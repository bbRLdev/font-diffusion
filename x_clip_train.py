from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from linformer import LinformerLM
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from constants import Constants 
from transformers import CLIPTokenizer
from vit_pytorch.efficient import ViT
from datasets import load_dataset
import argparse
import torch.nn as nn


from datasets import Image as HuggingFaceImage
from vit_train import get_vit_model
from vit_pytorch.extractor import Extractor

from transformers import CLIPTokenizer
from datasets import load_dataset
from datasets import Image as HuggingFaceImage
from linformer import Linformer
from vit_pytorch.efficient import ViT
import torch

from torch.optim import AdamW
from x_clip import CLIP
from vit_pytorch.extractor import Extractor
from torch.utils.data import DataLoader
import tqdm as tqdm

def get_tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
def prepare_data(tokenizer: CLIPTokenizer):
    def add_prompt(example):
        props = example['font_properties']
        character = example['character']
        split = character.split('_')
        if len(split) > 1:
            character = split[0] + 'case ' + split[1]
        else:
            character = split[0]
        prompt = f"a {props['font_serifs']} {character} with {props['width']} width {props['rounding']} corners {props['font_weight']} weight and {props['dynamics']} movement with characteristics that can be described by adjectives {example['font_characteristics']}" 
        example['prompt'] = prompt
        return example
    def map_tokens(example):
        prompt = example['prompt']
        tokens = tokenizer.encode(prompt, padding='max_length', max_length=42)
        example['tokens'] = tokens
        return example
    dataset = load_dataset('json', data_files={'train':'train-metadata.jsonl', 'test':'test-metadata.jsonl'})
    
    train_new_column = ['foo'] * len(dataset['train'])
    dataset['train'] = dataset['train'].add_column('prompt', train_new_column)
    dataset['train'] = dataset['train'].add_column('tokens', train_new_column)
    dataset['train'] = dataset['train'].map(add_prompt)
    dataset['train'] = dataset['train'].map(map_tokens)
    dataset['train'] = dataset['train'].remove_columns(['prompt', 'uniqueId', 'ttf_path', 'font_characteristics', 'font_properties', 'character', 'vit_label'])
    dataset['train'] = dataset['train'].cast_column('image', HuggingFaceImage())
    dataset['train'] = dataset['train'].with_format('torch')
    
    test_new_column = ['bar'] * len(dataset['test'])
    dataset['test'] = dataset['test'].add_column('prompt', test_new_column)
    dataset['test'] = dataset['test'].add_column('tokens', test_new_column)
    dataset['test'] = dataset['test'].map(add_prompt)
    dataset['test'] = dataset['test'].map(map_tokens)
    dataset['test'] = dataset['test'].remove_columns(['prompt', 'uniqueId', 'ttf_path', 'font_characteristics', 'font_properties', 'character', 'vit_label'])
    dataset['test'] = dataset['test'].cast_column('image', HuggingFaceImage())
    dataset['test'] = dataset['test'].with_format('torch')
    return dataset
def get_vit_model(image_size: int, patch_size: int, dim: int, depth: int, num_heads: int, k: int, device: str):
    sequence_length = (image_size//patch_size)**2 + 1
    # for 512x512px image with 32x32px patches: 16x16 + 1 CLS token
    efficient_transformer = Linformer(
        dim=dim,
        seq_len=sequence_length,  
        depth=depth,
        heads=num_heads,
        k=k
    )
    model = ViT(
        dim=dim,
        image_size=image_size,
        patch_size=patch_size,
        num_classes=62,
        transformer=efficient_transformer,
        channels=1,
    )
    return model 
def get_vit(image_size, patch_size, vit_dim, vit_depth, vit_num_heads, k, device, vit_checkpoint_path):
    vit = get_vit_model(image_size=image_size, 
                        patch_size=patch_size, 
                        dim=vit_dim, 
                        depth=vit_depth, 
                        num_heads=vit_num_heads, 
                        k=k, 
                        device=device)
    vit_checkpoint = torch.load(vit_checkpoint_path)
    if vit_checkpoint != None:
        vit.load_state_dict(vit_checkpoint['model_state_dict'])
        print('Loaded ViT model from checkpoint:', vit_checkpoint_path)
    return vit

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='x_clip_train.py')
    parser.add_argument('--vit_checkpoint', type=str, default=None, help='Path to the ViT checkpoint to load progress from')
    
    parser.add_argument('--patch_size', type=int, default=32, help='Desired image patch for ViT to create sequence of tokens. Must be divisible by image_size')
    parser.add_argument('--image_size', type=int, default=512, help='Size of training images.')
    parser.add_argument('--batch_size', type=int, default=8, help='Desired batch size.')
    parser.add_argument('--vit_dim', type=int, default=384, help='Last dimension of output tensor after linear transformation nn.Linear(..., dim).')
    parser.add_argument('--vit_linformer_k', type=int, default=64, help='k that the key/values are projected to along the sequence dimension')
    parser.add_argument('--vit_depth', type=int, default=12, help='Number of Transformer blocks.')
    parser.add_argument('--vit_num_heads', type=int, default=8, help='Number of heads to use in attention layers.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs to use.')
    parser.add_argument('--save_every_n_epochs', type=int, default=3, help='Save a checkpoint every n epochs')
    

    parser.add_argument('--valid_split', type=float, default=0.1, help='Percentage of images in train folder to use as validation while training ViT.')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate of ViT')
    parser.add_argument('--gamma', type=float, default=0.7, help='#TODO: Description needed')
    args = parser.parse_args()
    return args


class LinformerLM(nn.Module):
    def __init__(self, num_tokens, dim, seq_len, depth, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, reversible = False, dropout = 0.):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.linformer = Linformer(dim, seq_len, depth, k = k, heads = heads, dim_head = dim_head,
                one_kv_head = one_kv_head, share_kv = share_kv, reversible = reversible, dropout = dropout)
        # self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        x = self.linformer(x)
        # out = self.to_logits(x)
        return x

if __name__ == '__main__':
    args = _parse_args()
    device = 'cuda'
    clip_tokenizer = get_tokenizer()
    dataset = prepare_data(clip_tokenizer)
    print(dataset)
    def get_dataloaders(train_clip_dataset, test_clip_dataset, batch_size):
        train_loader = DataLoader(dataset=train_clip_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_clip_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    train_loader, valid_loader = get_dataloaders(train_dataset, test_dataset, 2)
    vit_checkpoint = './vit-checkpoints/model-epoch18.pt'
    image_size = 512
    patch_size = 32
    vit_dim = 128
    vit_depth = 12
    vit_num_heads = 8
    k = 64
    base_vit = get_vit(image_size, 
                        patch_size, 
                        vit_dim, 
                        vit_depth, 
                        vit_num_heads, 
                        k, 
                        device, 
                        vit_checkpoint)
    image_encoder = Extractor(
        base_vit,
        return_embeddings_only = True
    )
    text_encoder = LinformerLM(
        num_tokens=49408,
        dim = 256,
        seq_len = 42,
        depth = 12,
        heads = 8,
        dim_head = 64,        # be able to set the dimension of each head in multi-head attention
        k = 128,               # this is the k that the key/values are projected to along the sequence dimension
        one_kv_head = True,    # share one key/value head across all heads
        share_kv = False,      # share the same projection for keys and values
        reversible = False,      # make network reversible, like Reformer
    )
    clip = CLIP(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_image=128,
        dim_text=256,
        dim_latent=128,
        text_encode_without_mask=True,
        use_all_token_embeds=True,
        text_has_cls_token=False,
        visual_has_cls_token=True,
    ).to(device)
    def prepare_batch(batch):
        batch_imgs = batch['image']
        batch_tokens = batch['tokens']
        batch_imgs = batch_imgs[:, :, :, 0].unsqueeze(-1)
        batch_imgs = batch_imgs.permute(0, 3, 1, 2)
        batch_imgs = batch_imgs.type('torch.FloatTensor')
        return batch_imgs, batch_tokens
    for batch in train_loader:
        batch_imgs, batch_tokens = prepare_batch(batch)
        # batch_imgs.to(device)
        batch_tokens = batch_tokens.to(device)
        batch_imgs = batch_imgs.to(device)
        loss = clip(batch_tokens, batch_imgs, return_loss=True)
        loss.backward()
    lr=3e-5
    def get_trainable_params(model):
        return [params for params in model.parameters() if params.requires_grad]
    optimizer = AdamW(get_trainable_params(clip), lr=lr) # DALLE-pytorch setup
    for epoch in range(0, 10):
        epoch_loss = 0
        epoch_accuracy = 0
        for batch in tqdm(train_loader):
            batch_imgs, batch_tokens = prepare_batch(batch)
            # batch_imgs.to(device)
            batch_tokens = batch_tokens.to(device)
            batch_imgs = batch_imgs.to(device)
            loss = clip(batch_tokens, batch_imgs, return_loss=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # acc = (output.argmax(dim=1) == batch_labels).float().mean()
            # epoch_accuracy += acc / len(train_loader)
            # epoch_loss += loss / len(train_loader)
