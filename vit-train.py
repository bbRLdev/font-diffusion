from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from torchvision import transforms

from vit_pytorch.efficient import ViT
from datasets import load_dataset


from datasets import Image as HuggingFaceImage

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

def graph_random_sample(vit_dataset):
    _, axes = plt.subplots(3, 3, figsize=(12, 12))
    transform = transforms.ToPILImage()
    for _, ax in enumerate(axes.ravel()):
        r_idx = np.random.randint(len(vit_dataset), size=(1,))
        ex = vit_dataset[r_idx]
        img_tensor = ex['image']
        img_tensor = img_tensor.squeeze(0).permute(2, 0, 1)
        # print(ex['image'].squeeze(0).shape)
        img = transform(img_tensor)
        ax.set_title(ex['character'][0])
        ax.imshow(img)
def get_dataloaders(train_vit_dataset, test_vit_dataset, batch_size):
    train_loader = DataLoader(dataset=train_vit_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_vit_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def prepare_batch(batch):
    batch_imgs = batch['image']
    batch_labels = batch['character']
    batch_imgs = batch_imgs.permute(0, 3, 1, 2)
    batch_imgs = batch_imgs.type('torch.FloatTensor')
    return batch_imgs, batch_labels


dataset = load_dataset("json", data_files={'train': 'train-metadata.jsonl', 'test': 'test-metadata.jsonl'})
train_vit_dataset, test_vit_dataset = prepare_dataset_for_vit_training(dataset)
batch_size = 8
train_loader, test_loader = get_dataloaders(train_vit_dataset, test_vit_dataset, batch_size)
efficient_transformer = Linformer(
    dim=128,
    seq_len=256+1,  # 16x16 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)
device = 'cuda'
model = ViT(
    dim=128,
    image_size=512,
    patch_size=32,
    num_classes=62,
    transformer=efficient_transformer,
    channels=3,
).to(device)
# loss function
lr = 3e-5
gamma = 0.7
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(4):
    epoch_loss = 0
    epoch_accuracy = 0

    for batch in tqdm(train_loader):
        batch_imgs, batch_labels = prepare_batch(batch)
        print(batch_imgs.dtype, batch_labels.dtype)
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

    # with torch.no_grad():
    #     epoch_val_accuracy = 0
    #     epoch_val_loss = 0
    #     for data, label in valid_loader:
    #         data = data.to(device)
    #         label = label.to(device)

    #         val_output = model(data)
    #         val_loss = criterion(val_output, label)

    #         acc = (val_output.argmax(dim=1) == label).float().mean()
    #         epoch_val_accuracy += acc / len(valid_loader)
    #         epoch_val_loss += val_loss / len(valid_loader)

    # print(
    #     f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    # )