import argparse
import os
from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from dataset import get_dataloader
from model import VisionTransformer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
pl.seed_everything(42)

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--numgpus", default=1, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--epochs", default=180, type=int)
parser.add_argument("--ckpt", default='checkpoint', type=str)
parser.add_argument("--embed_dim", default=256, type=int)
parser.add_argument("--hidden_dim", default=512, type=int)
parser.add_argument("--num_heads", default=8, type=int)
parser.add_argument("--num_layers", default=6, type=int)
parser.add_argument("--patch_size", default=4, type=int)
parser.add_argument("--num_patches", default=64, type=int)
parser.add_argument("--drop_out", default=0.2, type=float)
parser.add_argument("--lr", default=3e-4, type=float)
args = parser.parse_args()
print(args)


class ViT(pl.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()

        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        # self.example_input_array = next(iter(tr))
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)

        return [optimizer], [lr_scheduler]
    
    def _calculate_loss(self, batch, mode='train'):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f'{mode}_loss', loss, prog_bar=True)
        self.log(f'{mode}_acc', acc, prog_bar=True)
        return loss
    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


def train_model(**kwargs):

    train_loader, val_loader, test_loader = get_dataloader(args.batch_size, args.workers)
    trainer = pl.Trainer(default_root_dir=os.path.join(args.ckpt, "ViT"),
                         gpus=args.numgpus,
                         max_epochs=args.epochs,
                         precision=16,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch"),
                                    TQDMProgressBar(refresh_rate=1)])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(args.ckpt, "ViT.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = ViT.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        model = ViT(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result

if __name__ == "__main__":
    
    model, results = train_model(model_kwargs={
        'embed_dim':args.embed_dim,
        'hidden_dim':args.hidden_dim,
        'num_heads':args.num_heads,
        'num_layers':args.num_layers,
        'patch_size':args.patch_size,
        'num_channels': 3,
        'num_patches': args.num_patches,
        'num_classes': 10,
        'dropout': args.drop_out
    }, lr=args.lr)

    print("Results", results)