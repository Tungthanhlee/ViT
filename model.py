import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

import pytorch_lightning as pl
pl.seed_everything(42)


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, drop_out=0.):
        super().__init__()

        self.layernorm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads) #should be self-implemented
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop_out),
        )

    def forward(self, x):

        inp_x = self.layernorm(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = self.layernorm(x)
        x = x + self.linear(x)

        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.):
        super().__init__()

        self.patch_size = patch_size

        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, drop_out=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self.drop_out = nn.Dropout(dropout)

        #Parameters
        self.cls_token = nn.Parameter(torch.rand(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.rand(1, 1+num_patches, embed_dim))
    
    def forward(self, x):
        #Process input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        #Add CLS
        cls_token = self.cls_token.repeat(B, 1,1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[: , :T+1]

        #apply transformer
        x = self.drop_out(x)
        x = x.transpose(0,1)
        x = self.transformer(x)

        #perform transformer
        cls = x[0]
        out = self.mlp_head(cls)

        return out


