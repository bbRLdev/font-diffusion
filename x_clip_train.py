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

from datasets import Image as HuggingFaceImage
from vit_train import get_vit_model
from vit_pytorch.extractor import Extractor

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
        example['tokens'] = tokenizer.encode(prompt, padding='longest')
        return example
    dataset = load_dataset('json', data_files={'train':'train-metadata.jsonl', 'test':'test-metadata.jsonl'})
    train_new_column = ['foo'] * len(dataset['train'])
    dataset['train'] = dataset['train'].add_column('prompt', train_new_column)
    test_new_column = ['bar'] * len(dataset['test'])
    dataset['test'] = dataset['test'].add_column('prompt', test_new_column)
    dataset['train'] = dataset['train'].map(add_prompt)
    dataset['test'] = dataset['test'].map(add_prompt)
    dataset['train'] = dataset['train'].add_column('tokens', ['foo'] * len(dataset['train']))
    dataset['test'] = dataset['test'].add_column('tokens', ['foo'] * len(dataset['test']))
    dataset['test']['tokens'] = tokenizer.encode(dataset['test'].select_columns('prompts'), padding='longest')
    dataset['train']['tokens'] = tokenizer.encode(dataset['train'].select_columns('prompts'), padding='longest')

    # dataset['train'] = dataset['train'].map(map_tokens)
    # dataset['test'] = dataset['test'].map(map_tokens)
    return dataset

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
    device = 'cuda'
    model = ViT(
        dim=dim,
        image_size=image_size,
        patch_size=patch_size,
        num_classes=Constants.NUM_CLASSES,
        transformer=efficient_transformer,
        channels=1,
    ).to(device)
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

if __name__ == '__main__':
    args = _parse_args()
    device = 'cuda'

    vit_checkpoint = args.vit_checkpoint
    image_size = args.image_size
    patch_size = args.patch_size
    vit_dim = args.vit_dim
    vit_depth = args.vit_depth
    vit_num_heads = args.vit_num_heads
    k = args.vit_linformer_k

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
    clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    model = LinformerLM(
        num_tokens = 20000,
        dim = 512,
        seq_len = 32,
        depth = 12,
        heads = 8,
        dim_head = 128,        # be able to set the dimension of each head in multi-head attention
        k = 256,               # this is the k that the key/values are projected to along the sequence dimension
        one_kv_head = True,    # share one key/value head across all heads
        share_kv = False,      # share the same projection for keys and values
        reversible = True      # make network reversible, like Reformer
    )

