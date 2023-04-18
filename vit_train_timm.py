import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from constants import Constants 
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from datasets import load_dataset
import argparse

from datasets import Image as HuggingFaceImage

def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='find-camera-values.py')

    parser.add_argument('--from_checkpoint', type=str, default=None, required=False, help='Path to the checkpoint to load progress from')
    parser.add_argument('--model_name', type=str, default='mobilevitv2_050', help='Model name to load from')
    parser.add_argument('--valid_split', type=float, default=0.1, help='Percentage of images in train folder to use as validation while training ViT.')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate of ViT')
    parser.add_argument('--save_every_n_epochs', type=int, default=3, help='save checkpoint every n epochs')
    parser.add_argument('--gamma', type=float, default=0.7, help='#TODO: Description needed')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--run_name', type=str, required=True, help='Checkpoint path name for this training session')
    args = parser.parse_args()
    return args
def prepare_dataset_for_vit_training(dataset):
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    train_vit_imgs_only = train_dataset.remove_columns(['uniqueId', 'ttf_path', 'font_characteristics', 'font_properties', 'character'])
    train_vit_imgs_only = train_vit_imgs_only.class_encode_column("vit_label")
    train_vit_imgs_only = train_vit_imgs_only.cast_column('image', HuggingFaceImage())
    train_vit_imgs_only.set_format(type='numpy', columns=['image', 'vit_label'])

    test_vit_imgs_only = test_dataset.remove_columns(['uniqueId', 'ttf_path', 'font_characteristics', 'font_properties', 'character'])
    test_vit_imgs_only = test_vit_imgs_only.class_encode_column("vit_label")
    test_vit_imgs_only = test_vit_imgs_only.cast_column('image', HuggingFaceImage())
    test_vit_imgs_only.set_format(type='numpy', columns=['image', 'vit_label'])
    return train_vit_imgs_only, test_vit_imgs_only

def get_dataloaders(train_vit_dataset, valid_vit_dataset, test_vit_dataset, batch_size):
    train_loader = DataLoader(dataset=train_vit_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_vit_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_vit_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader

def apply_transform_to_numpy_images(transform, images_array):
    images_array = images_array.numpy()
    transformed_images = []
    for img in images_array:
        transformed = transform(image=img)
        transformed_images.append(transformed['image'])

    return torch.stack(transformed_images)
def save_model(model_folder, enc, epoch):
    if not os.path.exists(model_folder): 
        os.makedirs(model_folder)
    torch.save(enc.state_dict(), os.path.join(model_folder, f"enc_{epoch}.pt"))
def prepare_batch(batch):
    batch_imgs = batch['image']
    transform = A.Compose([
        A.Resize(256, 256, interpolation=1, always_apply=True, p=1),
        ToTensorV2()
    ])
    batch_imgs = apply_transform_to_numpy_images(transform, batch_imgs)
    batch_labels = batch['vit_label']
    batch_imgs = batch_imgs[:, :, :, 0].unsqueeze(-1)
    batch_imgs = batch_imgs.permute(0, 3, 1, 2) / 255
    return batch_imgs, batch_labels

def get_vit_model(model_name, num_classes=62, in_channels=1):
    model = timm.create_model(model_name, in_chans=in_channels, pretrained=True)
    model.reset_classifier(num_classes)
    model = model.to('cuda')
    
    return model

# def get_optimizer_criterion_and_scheduler(lr, gamma):

#     return criterion, optimizer, scheduler


if __name__ == '__main__':
    if not os.path.exists(Constants.VIT_CHECKPOINTS_PATH):
        os.makedirs(Constants.VIT_CHECKPOINTS_PATH)
    args = _parse_args()
    
    # prepare dataset, dataloaders
    dataset = load_dataset("json", data_files={'train': 'train-metadata.jsonl', 'test': 'test-metadata.jsonl'})
    train_vit_dataset, test_vit_dataset = prepare_dataset_for_vit_training(dataset)
    train_vit_dataset = train_vit_dataset.train_test_split(test_size=args.valid_split)
    batch_size = args.batch_size
    train_loader, valid_loader, test_loader = get_dataloaders(train_vit_dataset['train'], train_vit_dataset['test'], test_vit_dataset, batch_size)
    device = 'cuda'
    model = get_vit_model(model_name=args.model_name, num_classes=Constants.NUM_CLASSES, in_channels=Constants.NUM_RGB_CHANNELS)
    lr, gamma = args.learning_rate, args.gamma
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)    
    
    start_epoch = 0
    # load model checkpoint if necessary
    if not args.from_checkpoint is None:
        model.load(args.checkpoint, n_classes=args.N_classes)

    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for batch in tqdm(train_loader):
            batch_imgs, batch_labels = prepare_batch(batch)
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)
            output = model(batch_imgs)
            # print(output.shape)
            loss = criterion(output, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (output.argmax(dim=1) == batch_labels).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            model.eval()
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
        if epoch % args.save_every_n_epochs == 0:
            save_model(args.run_name, model, epoch=epoch)

    