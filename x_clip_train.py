import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig
from vit_pytorch.efficient import ViT
from datasets import load_dataset
import argparse
import torch.nn as nn

from datasets import load_dataset
from datasets import Image as HuggingFaceImage
from vit_train import get_vit_model
from vit_pytorch.extractor import Extractor

from linformer import Linformer

from x_clip import CLIP
from vit_pytorch.extractor import Extractor
from torch.utils.data import DataLoader
from CLIPConstants import CLIPConstants
import os

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
    
def get_tokenizer(add_special_tokens) -> CLIPTokenizer:
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    if add_special_tokens:
        special_tokens = {'mask_token':CLIPConstants.MASK_TOKEN}
        print('Added special tokens: ', special_tokens)
        tokenizer.add_special_tokens(special_tokens)
    return tokenizer
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
def get_dataloaders(train_clip_dataset, test_clip_dataset, batch_size):
    train_loader = DataLoader(dataset=train_clip_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_clip_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
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
def get_vit(image_size, patch_size, vit_dim, vit_depth, vit_num_heads, k, device, checkpoint_path):
    vit = get_vit_model(image_size=image_size, 
                        patch_size=patch_size, 
                        dim=vit_dim, 
                        depth=vit_depth, 
                        num_heads=vit_num_heads, 
                        k=k, 
                        device=device)
    vit_checkpoint = torch.load(checkpoint_path)
    if vit_checkpoint != None:
        vit.load_state_dict(vit_checkpoint['model_state_dict'])
        print('Loaded ViT model from checkpoint:', checkpoint_path)
    return vit
def prepare_batch(batch):
    batch_imgs = batch['image']
    batch_tokens = batch['tokens']
    batch_imgs = batch_imgs[:, :, :, 0].unsqueeze(-1)
    batch_imgs = batch_imgs.permute(0, 3, 1, 2)
    batch_imgs = batch_imgs.type('torch.FloatTensor')
    return batch_imgs, batch_tokens
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
    parser.add_argument('--vit_dim', type=int, default=128, help='Last dimension of output tensor after linear transformation nn.Linear(..., dim).')
    parser.add_argument('--vit_linformer_k', type=int, default=64, help='k that the key/values are projected to along the sequence dimension')
    parser.add_argument('--vit_depth', type=int, default=12, help='Number of Transformer blocks.')
    parser.add_argument('--vit_num_heads', type=int, default=8, help='Number of heads to use in attention layers.')
    
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate of ViT')
    parser.add_argument('--gamma', type=float, default=0.7, help='#TODO: Description needed')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs to use.')
    parser.add_argument('--save_every_n_epochs', type=int, default=5, help='Save a checkpoint every n epochs')
    
    parser.add_argument('--text_encoder_dim', type=int, default=512, help='Output dimension of the text encoder')
    parser.add_argument('--text_encoder_max_seq_len', type=int, default=42, help='Maximum token input sequence length')
    parser.add_argument('--use_pretrained_text_encoder', type=bool, default=False)
    parser.add_argument('--pretrained_text_encoder_name', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--text_encoder_depth', type=int, default=12, help='Depth of text encoder')
    parser.add_argument('--text_encoder_num_heads', type=int, default=8, help='Number of heads for text encoder')
    parser.add_argument('--text_encoder_dim_head', type=int, default=64, help='Number of heads for text encoder')
    parser.add_argument('--text_encoder_k_projection', type=int, default=128, help='Dimension for LinformerLM to project to')

    parser.add_argument('--clip_latent_dim', type=int, default=384, help='CLIP latent dimension projection dim')
    parser.add_argument('--use_mlm', type=bool, default=False, help='Use MLM (DECLIP)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    CHECKPOINTS_PATH = os.path.join(os.getcwd(), 'clip-checkpoints')
    if not os.path.exists(CHECKPOINTS_PATH):
        os.mkdir(CHECKPOINTS_PATH)
    args = _parse_args()
    device = 'cuda'
    use_mlm = args.use_mlm
    clip_tokenizer = get_tokenizer(use_mlm)
    print("special_tokens: ", clip_tokenizer.all_special_tokens, clip_tokenizer.all_special_ids)
    dataset = prepare_data(clip_tokenizer)   
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    batch_size = args.batch_size
    train_loader, valid_loader = get_dataloaders(train_dataset, test_dataset, batch_size)
    base_vit = get_vit(
        image_size=args.image_size, 
        patch_size=args.patch_size, 
        vit_dim = args.vit_dim,
        vit_depth = args.vit_depth,
        vit_num_heads = args.vit_num_heads,
        k = args.vit_linformer_k,
        device=device, 
        checkpoint_path=args.vit_checkpoint
    )
    image_encoder = Extractor(
        base_vit,
        return_embeddings_only = True
    )
    if args.use_pretrained_text_encoder:
        model_name = args.pretrained_text_encoder_name
        text_encoder = CLIPTextModel.from_pretrained(model_name)
        text_encoder.config.bos_token_id = clip_tokenizer.bos_token_id
        text_encoder.config.pad_token_id = clip_tokenizer.pad_token_id
        text_encoder.config.eos_token_id = clip_tokenizer.eos_token_id
        text_encoder_dim = text_encoder.config.projection_dim
        print('text_encoder_dim: ', text_encoder_dim)
        text_encoder.resize_token_embeddings(len(clip_tokenizer))
        clip = CLIP(
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            dim_image=args.vit_dim,
            dim_text=text_encoder_dim,
            dim_latent=args.clip_latent_dim,
            text_encode_without_mask=False,
            use_all_token_embeds=False,
            text_has_cls_token=True,
            visual_has_cls_token=True,
            num_text_tokens=text_encoder.vocab_size,
            text_pad_id=clip_tokenizer.pad_token_id,
            text_eos_id=clip_tokenizer.eos_token_id,
            use_mlm=True,
            mlm_mask_token_id=clip_tokenizer.mask_token_id,
            mlm_pad_token_id=clip_tokenizer.pad_token_id,
            mlm_mask_ignore_token_ids=[clip_tokenizer.bos_token_id]
        ).to(device)
    else:
        text_encoder_dim = args.text_encoder_dim
        text_encoder = LinformerLM(
            num_tokens=clip_tokenizer.vocab_size,
            dim = text_encoder_dim,
            seq_len = args.text_encoder_max_seq_len,
            depth = args.text_encoder_depth,
            heads = args.text_encoder_num_heads,
            dim_head = args.text_encoder_dim_head,        # be able to set the dimension of each head in multi-head attention
            k = args.text_encoder_k_projection,               # this is the k that the key/values are projected to along the sequence dimension
            one_kv_head = True,    # share one key/value head across all heads
            share_kv = False,      # share the same projection for keys and values
            reversible = False,      # make network reversible, like Reformer
        )
        clip = CLIP(
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            dim_image=args.vit_dim,
            dim_text=text_encoder_dim,
            dim_latent=args.clip_latent_dim,
            text_encode_without_mask=True,
            use_all_token_embeds=False,
            text_has_cls_token=False,
            visual_has_cls_token=True,
            use_mlm=True,
            num_text_tokens=clip_tokenizer.vocab_size,
            text_ssl_loss_weight=0.5
        ).to(device)

    num_epochs = args.num_epochs
    lr = args.learning_rate
    def get_trainable_params(model):
        return [params for params in model.parameters() if params.requires_grad]
    optimizer = AdamW(get_trainable_params(clip), lr=lr) # DALLE-pytorch setup
    for epoch in range(0, num_epochs):
        epoch_loss = 0
        # epoch_accuracy = 0
        for batch in tqdm(train_loader):
            batch_imgs, batch_tokens = prepare_batch(batch)
            # batch_imgs.to(device)
            batch_tokens = batch_tokens.to(device)
            batch_imgs = batch_imgs.to(device)
            loss = clip(batch_tokens, batch_imgs, return_loss=True, freeze_image_encoder=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            # acc = (output.argmax(dim=1) == batch_labels).float().mean()
            # epoch_accuracy += acc / len(train_loader)
            # epoch_loss += loss / len(train_loader)
        with torch.no_grad():
            valid_loss = 0.0
            for batch in valid_loader:
                batch_imgs, batch_tokens = prepare_batch(batch)
                batch_imgs = batch_imgs.to(device)
                batch_tokens = batch_tokens.to(device)
                loss = clip(batch_tokens, batch_imgs, return_loss=True)
                valid_loss += loss
        print(f'Epoch {epoch+1} train loss: {epoch_loss}, Epoch average train loss: {epoch_loss/len(train_dataset)}')
        print(f'Epoch {epoch+1}  valid loss: {valid_loss}, Epoch average valid loss: {valid_loss/len(test_dataset)}')
        if (epoch + 1) % args.save_every_n_epochs == 0:
            save_path = os.path.join(CHECKPOINTS_PATH, f'clip-epoch-{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': clip.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, save_path)
            print(f'CLIP checkpoint for epoch {epoch} saved at: {save_path}')
