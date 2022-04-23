import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.utils.data as data

import pytorch_lightning as pl
pl.seed_everything(42)
    
def get_dataset(download=False):
    train_trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                     ])
    test_trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                    ])      

    train_dataset = CIFAR10(root='data/', train=True, transform=train_trans, download=download)
    val_dataset = CIFAR10(root='data/', train=True, transform=test_trans, download=download)
    pl.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    pl.seed_everything(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR10(root='data/', train=False, transform=test_trans, download=download)

    return train_set, val_set, test_set

def get_dataloader(batch_size, num_workers):
    train_set, val_set, test_set = get_dataset(False)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    NUM_IMAGES = 8
    DOWNLOAD=True
    train_set, val_set, test_set = get_dataset(DOWNLOAD)
    CIFAR_images = torch.stack([val_set[idx][0] for idx in range(NUM_IMAGES)], dim=0)
    img_grid = torchvision.utils.make_grid(CIFAR_images, nrow=4, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)
    plt.figure(figsize=(8,8))
    plt.title("Image examples of the CIFAR10 dataset")
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()
    plt.close()
