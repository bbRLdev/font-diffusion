from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from constants import Constants 

from vit_pytorch.efficient import ViT
from datasets import load_dataset
import argparse

from datasets import Image as HuggingFaceImage
from vit_train import get_vit_model
from vit_pytorch.extractor import Extractor

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='x_clip_train.py')
    parser.add_argument('--vit_checkpoint', type=str, required=True, default=None, required=False, help='Path to the ViT checkpoint to load progress from')
    
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
def get_vit(image_size, patch_size, vit_dim, vit_depth, vit_num_heads, k, device, vit_checkpoint):
    vit = get_vit_model(image_size=image_size, 
                        patch_size=patch_size, 
                        dim=vit_dim, 
                        depth=vit_depth, 
                        num_heads=vit_num_heads, 
                        k=k, 
                        device=device)
    vit_checkpoint = torch.load(vit_checkpoint)
    vit.load_state_dict(vit_checkpoint['model_state_dict'])
    print('Loaded model from checkpoint:', vit_checkpoint)
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

    vit = get_vit(image_size, patch_size, vit_dim, vit_depth, vit_num_heads, k, device, vit_checkpoint)

    image_encoder = Extractor(
        vit,
        return_embeddings_only = True
    )
