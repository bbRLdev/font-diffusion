from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchvision import transforms

from vit_pytorch.efficient import ViT
from datasets import load_dataset
import argparse

from datasets import Image as HuggingFaceImage

from datasets import Image as HuggingFaceImage
def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='find-camera-values.py')
    parser.add_argument('--patch_size', type=int, default=32, help='Desired image patch for ViT to create sequence of tokens. Must be divisible by image_size')
    parser.add_argument('--image_size', type=int, default=512, help='Size of training images.')
    parser.add_argument('--batch_size', type=int, default=8, help='Desired batch size.')
    parser.add_argument('--transformer_dim', type=int, default=128, help='Last dimension of output tensor after linear transformation nn.Linear(..., dim).')
    parser.add_argument('--transformer_depth', type=int, default=12, help='Number of Transformer blocks.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads to use in attention layers.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs to use.')

    parser.add_argument('--valid_split', type=float, default=0.1, help='Percentage of images in train folder to use as validation while training ViT.')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate of ViT')
    parser.add_argument('--gamma', type=float, default=0.7, help='#TODO: Description needed')
    args = parser.parse_args()
    return args
def prepare_dataset_for_vit_training(dataset):
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    train_vit_imgs_only = train_dataset.remove_columns(['uniqueId', 'ttf_path', 'font_characteristics', 'font_properties'])
    train_vit_imgs_only = train_vit_imgs_only.class_encode_column("character")
    train_vit_imgs_only = train_vit_imgs_only.cast_column('image', HuggingFaceImage())
    train_vit_imgs_only = train_vit_imgs_only.with_format('torch')

    test_vit_imgs_only = test_dataset.remove_columns(['uniqueId', 'ttf_path', 'font_characteristics', 'font_properties'])
    test_vit_imgs_only = test_vit_imgs_only.class_encode_column("character")
    test_vit_imgs_only = test_vit_imgs_only.cast_column('image', HuggingFaceImage())
    test_vit_imgs_only = test_vit_imgs_only.with_format('torch')
    return train_vit_imgs_only, test_vit_imgs_only

def get_dataloaders(train_vit_dataset, valid_vit_dataset, test_vit_dataset, batch_size):
    train_loader = DataLoader(dataset=train_vit_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_vit_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_vit_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader
def prepare_batch(batch):
    batch_imgs = batch['image']
    batch_labels = batch['character']
    batch_imgs = batch_imgs.permute(0, 3, 1, 2)
    batch_imgs = batch_imgs.type('torch.FloatTensor')
    return batch_imgs, batch_labels

if __name__ == '__main__':
    args = _parse_args()
    dataset = load_dataset("json", data_files={'train': 'train-metadata.jsonl', 'test': 'test-metadata.jsonl'})
    train_vit_dataset, test_vit_dataset = prepare_dataset_for_vit_training(dataset)
    train_vit_dataset = train_vit_dataset.train_test_split(test_size=args.valid_split)
    batch_size = args.batch_size
    train_loader, valid_loader, test_loader = get_dataloaders(train_vit_dataset['train'], train_vit_dataset['test'], test_vit_dataset, batch_size)
    sequence_length = (args.image_size//args.patch_size)**2 + 1
    # for 512x512px image with 32x32px patches: 16x16 + 1 CLS token
    efficient_transformer = Linformer(
        dim=args.transformer_dim,
        seq_len=sequence_length,  
        depth=args.transformer_depth,
        heads=args.num_heads,
        k=64
    )
    device = 'cuda'
    model = ViT(
        dim=args.transformer_dim,
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=62,
        transformer=efficient_transformer,
        channels=3,
    ).to(device)
    # loss function
    lr = args.learning_rate
    gamma = args.gamma
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(args.num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for batch in tqdm(train_loader):
            batch_imgs, batch_labels = prepare_batch(batch)
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            output = model(batch_imgs)
            loss = criterion(output, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == batch_labels).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for batch in valid_loader:
                batch_imgs, batch_labels = prepare_batch(batch)
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                val_output = model(batch_imgs)
                val_loss = criterion(val_output, batch_labels)

                acc = (val_output.argmax(dim=1) == batch_labels).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - \
                val_acc: {epoch_val_accuracy:.4f}\n"
        )
