# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 20:33:44 2025

@author: User
"""
import argparse
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import torch.nn as nn
import numpy as np
import sys, os
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import pickle
import random
import scipy
import time 
import gc
import json

from copy import deepcopy
from torch.utils import data
from torch.utils.data import ConcatDataset
from torch.nn import functional as F
from multiprocessing import Queue, Process
from PIL import Image

from lib_train import *
from lib_cnn import * 
from lib_IGviz import *
from lib_influence_groundtruth import * 

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


def genloaders_vision_test(loader_params, labelnoise_params, image_size=(224, 224)):

    def preprocess_dataset(dataset, is_grayscale=False):
        
        if isinstance(dataset.data, np.ndarray):
            data = torch.from_numpy(dataset.data)
        else:
            data = dataset.data  # Already a torch.Tensor

        data = data.float() / 255.0

        if is_grayscale:
            data = data.unsqueeze(1)  # Add channel dim for grayscale: [N, 1, H, W]
        else:
            data = data.permute(0, 3, 1, 2)  # [N, H, W, C] → [N, C, H, W]

        targets = torch.tensor(dataset.targets)
        return data_resized, targets

    def preprocess_dataset_from_imagefolder(dataset):
        data = []
        targets = []

        for img, label in dataset:
            data.append(img)
            targets.append(label)

        data = torch.stack(data)
        targets = torch.tensor(targets)
        return data, targets

    dataset_name = loader_params['dataset_name']
    root         = loader_params['root_folder']
    is_grayscale = False

    # Simplify for grayscale datasets
    if dataset_name in ['MNIST', 'FashionMNIST']:
        is_grayscale = True
    
    ############################## Load relevant datasets ##############################
    
    if dataset_name == 'FashionMNIST':
        data_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=data_transform)
        test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=data_transform)
    
    elif dataset_name == 'MNIST':
        data_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=data_transform)
        test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=data_transform)

    ############################## Additional fine grained datasets ##############################
    
    elif dataset_name == 'Flowers102':
        flowers_train = torchvision.datasets.Flowers102(
            root=root, split='train', download=True,
            transform=transforms.Compose([
                torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms(),
            ])
        )
        flowers_val   = torchvision.datasets.Flowers102(
            root=root, split='val', download=True,
            transform=transforms.Compose([
                torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms(),
            ])
        )
        flowers_test  = torchvision.datasets.Flowers102(
            root=root, split='test', download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomAffine(degrees=30, shear=20),
                # Resize image and normalize pixels using the provided mean and standard deviation
                torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms(),
            ])
        )
        # Use original test split as new train dataset (6149 images)
        train = flowers_test
        # Combine train + val into one test dataset (2040 images)
        test = ConcatDataset([flowers_train, flowers_val])

    elif dataset_name == 'FGVCAircraft':
        train = torchvision.datasets.FGVCAircraft(
            root=root, split='trainval', download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomAffine(degrees=30, shear=20),
                # Resize image and normalize pixels using the provided mean and standard deviation
                torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms(),
            ])
        )
        test = torchvision.datasets.FGVCAircraft(
            root=root, split='test', download=True,
            transform=transforms.Compose([
                torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms(),
            ])
        )

    elif dataset_name == 'SVHN':
        train = torchvision.datasets.SVHN(root=root, split='train', download=True)
        test = torchvision.datasets.SVHN(root=root, split='test', download=True)
        extra = torchvision.datasets.SVHN(root=root, split='extra', download=True)

    else:
        print("ERROR: Unknown dataset:", dataset_name)


    dataset_data, dataset_targets = None, None
    dataset_test_data, dataset_test_targets = None, None
    
    if dataset_name in ['FashionMNIST', 'MNIST', 'CIFAR10', 'CIFAR100']:
        dataset_data, dataset_targets = preprocess_dataset(
            train,
            is_grayscale=is_grayscale
        )
        dataset_test_data, dataset_test_targets = preprocess_dataset(
            test,
            is_grayscale=is_grayscale
        )
    else:
        dataset_data, dataset_targets = preprocess_dataset_from_imagefolder(train)
        dataset_test_data, dataset_test_targets = preprocess_dataset_from_imagefolder(test)

    ############################## Apply Label Noise ##############################
    if labelnoise_params['noise_type'] is not None and labelnoise_params['noise_level'] > 0.0:
        num_classes = len(torch.unique(dataset_targets))
        dataset_targets = add_label_noise(
            dataset_targets,
            labelnoise_params['noise_type'], 
            labelnoise_params['noise_level'],
            num_classes
        )

    ############################## Generate DataLoaders ##############################
    trainloader, testloader, IG_trainloader = genloaders(
        dataset_data,
        dataset_targets,
        dataset_test_data,
        dataset_test_targets,
        loader_params
    )
        
    return trainloader, testloader, IG_trainloader



def prerequisites():
    # -------------- Seed Setup --------------
    gc.collect()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.empty_cache()


logger = None

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    ############################## Argument Parser ##############################
    parser = argparse.ArgumentParser(description="Run influence estimation with label noise")
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='ShallowMNIST', help='Model for experiment')
    parser.add_argument('--root_folder', type=str, default='../data', help='Root folder for data')
    parser.add_argument('--noise_type', type=str, default='symmetric', choices=['symmetric', 'asymmetric', 'none'], help='Type of label noise')
    parser.add_argument('--program_mode', type=str, default='normal', choices=['normal', 'GT'], help='Run mode')
    parser.add_argument('--save_mode', type=str, default='load', choices=['load', 'store', 'none'], help='Save or load influence graph')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization of influence pairs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for loaders')
    parser.add_argument('--img_size', type=int, default=32, help='Size to resize input image to')
    parser.add_argument('--log_verbosity', type=int, default=1, help='log message verbosity (0=critical, 1=info, 2=debug)')
    args = parser.parse_args()

    # -------------- Start logger queue and listener process --------------
    log_queue = Queue()
    listener = Process(target=log_listener, args=(log_queue, 1))
    listener.start()

    logger = InfluenceLogger(log_queue, verbose=args.log_verbosity)
    logger.log("Main process started.", level=1)

    # -------------- Unpack parser arguments --------------
    dataset      = args.dataset
    model_name   = args.model_name
    root_folder  = args.root_folder
    program_mode = args.program_mode # normal or GT (Ground truth)
    save_mode    = args.save_mode # store, load or none
    visualize    = args.visualize
    noise_type   = args.noise_type
    noise_level  = 0
    num_workers  = args.num_workers
    img_size     = args.img_size

    # -------------- Device Setup --------------
    DEVICE = torch.device(args.device)
    if not torch.cuda.is_available():
        logger.log("WARN: Cuda unavailable, defaulting to cpu...", level=0)
        DEVICE = torch.device('cpu')
    logger.log("Using device: {}".format(DEVICE), level=1)

    prerequisites()

    labelnoise_params = {
        'noise_type':  noise_type,
        'noise_level': noise_level
    }
    loader_params = {
        'dataset_name':     dataset,
        'conversion':       'none',
        'root_folder':      root_folder,
        'training_size':    'full', # 'full'
        'batch_size':       32,   # 20-40
        'IG_batch_size':    400, 
        'transform':        None,
        'add_singleton':    False,
        'convert_to_torch': False,
        'num_workers':      num_workers,
    }
    influence_params = {
        'loss_scaling_span':  'full', # 'batch' or 'full'
        'loss_scaling_type':  'root_mean_squared', # 'mean' or 'mean_absolute' or None
        'set_zero_mean':      False, # 'full' or 'separate'
        'class_normalize' :   False,
        'remove_negatives' :  False,
        'clipping' :          False,
        'intraclass_only' :   True,
        'negative_clipping':  False,
        'clip_outliers':      False,
        'mode':               'mean', # For InfluenceGraphv3
        'gradient_lr':        0.1, # For InfluenceGraphv3
        'dtype':              np.float32,
        'graph_type':         InfluenceGraphv4,
    }
    train_params = {
        'optimizer':           'Adam',
        'init_rate':           1e-3,
        'total_epochs':        20,
        'weight_decay':        1e-4,
        'scheduler': {
            'name':            'StepLR',
            'step_size':       6,
            'gamma':           0.3
        },
        'criterion':           'CrossEntropyLoss',
        'disp_epoch':          True,
        'disp_loss_epoch':     False,
        'disp_time_per_epoch': True, 
        'disp_loss_final':     True, 
        'disp_accuracy_final': True
    }
    influence_GT_params = {
        'type':                'batch', # batch or representative
        'training_iterations': train_params['total_epochs'],
        'intraclass_only':     True,
        'dtype':               np.float32,
    }
    influence_GT_train_params = {
        'optimizer':           'SGD',
        'scheduler':           {'name': None}, # 'step_size': 10, 'milestones':[10,20,30],'gamma':0.8, 'max_lr': 0.01}
        'init_rate':           0.1,
        'total_epochs':        50,
        'weight_decay':        0, 
        'criterion':           'CrossEntropyLoss',
        'disp_epoch':          False,
        'disp_loss_epoch':     True,
        'disp_time_per_batch': True,
        'disp_total_time':     True,
    }
    model_params = {
        'type':                CNN,
        'name':                model_name,
        'in_channels':         1,
        'num_classes':         10,
        'img_size':            img_size,
        'batchnorm':           True,
    }
    
    
    # -------------- Customize arguments based on model --------------
    if dataset == 'MNIST' or dataset == 'FashionMNIST':
        model_params['in_channels'] = 1
        model_params['num_classes'] = 10
    
    elif dataset == 'CIFAR10':
        model_params['in_channels'] = 3
        model_params['num_classes'] = 10
    
    elif dataset == 'Flowers102':
        model_params['in_channels'] = 3
        model_params['num_classes'] = 102
    
    if model_params['name'] == 'pretrained_VGG16':
        model = get_pretrained_vgg16(num_classes=model_params['num_classes'], fine_tune='NEW_LAYERS')
    else:
        model = model_params['type'](
            model_params['name'],
            in_channels = model_params['in_channels'],
            num_classes = model_params['num_classes'],
            img_size    = model_params['img_size'],
            batchnorm   = model_params['batchnorm']
        )
        

    image_size = (img_size, img_size)
    trainloader, testloader, IG_trainloader = genloaders_vision_test(
        loader_params,
        labelnoise_params,
        image_size=image_size
    )

    print( 'Initial train accuracy:', test_model(model, trainloader) )
    print( 'Initial test accuracy:', test_model(model, testloader) )

    model = train_model_general(
        model,
        trainloader,
        train_params,
        # config=model_params['name'],
        config=None,
        logger=logger
    )
    
    print( 'Train accuracy:', test_model(model, trainloader) )
    print( 'Test accuracy:', test_model(model, testloader) )

    # -------------- Stop logger listener cleanly, and cleanup --------------
    logger.log("Main process done.", level=1)
    logger.log("Shutting down logger listener...", level=1)
    log_queue.put(None)  # Sentinel to signal listener shutdown
    listener.join()

    gc.collect()
    torch.cuda.empty_cache()



