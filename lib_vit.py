# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:47:21 2025

@author: User
"""

import torch
from vit_pytorch import SimpleViT

vit_mnist ={
    "image_size": 28,
    "patch_size": 4,
    "dim": 64,
    "heads": 4,
    "depth": 6,
    "dropout": 0.1,
    "mlp_dim": 128, # forward_mul x dim (embed_dim)
    }

def create_vit(vit_name, channels=1, num_classes=10):
   
    v = SimpleViT(
        image_size = vit_name["image_size"],
        patch_size = vit_name["image_size"],
        num_classes = num_classes,
        dim = vit_name["image_size"],
        depth = vit_name["image_size"],
        heads = vit_name["image_size"],
        channels = channels, 
        mlp_dim = vit_name["image_size"]
    )

img = torch.randn(1, 3, 256, 256)