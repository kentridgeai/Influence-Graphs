# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:02:52 2025

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:55:40 2025

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



def genloaders_vision(loader_params, image_size=(224, 224), logger=None):

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
        return data, targets

    def preprocess_dataset_from_imagefolder(dataset, save_path=None):
        data = []
        targets = []
    
        for img, label in dataset:
            data.append(img)
            targets.append(label)
    
        data = torch.stack(data)
        targets = torch.tensor(targets)
    
        if save_path is not None:
            torch.save({'data': data, 'targets': targets}, save_path)
            print(f"✅ Saved preprocessed dataset to {save_path}")
    
        return data, targets
        
    def load_preprocessed_dataset(save_path):
        saved = torch.load(save_path)
        print(f"✅ Loaded preprocessed dataset from {save_path}")
        return saved['data'], saved['targets']

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

    elif dataset_name == 'CIFAR10':
        data_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=data_transform)
        test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=data_transform)

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
        logger.log("ERROR: Unknown dataset {}.".format(dataset_name), level=0)


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
        if dataset_name not in ['FGVCAircraft', 'Flowers102']:
            logger.log("ERROR: Unknown dataset {} identified, unable to preprocess data...".format(dataset_name), level=0)
            
        save_train_path = 'train_flowers102.pth' if dataset_name == 'Flowers102' else 'train_fgvcaircraft.pth'
        save_train_path = os.path.join(dir_path, save_train_path)
        
        save_test_path  = 'test_flowers102.pth'  if dataset_name == 'Flowers102' else 'test_fgvcaircraft.pth'
        save_test_path  = os.path.join(dir_path, save_test_path)

        if os.path.exists(save_train_path) and os.path.exists(save_test_path):
            # Load preprocessed files
            dataset_data, dataset_targets = load_preprocessed_dataset(save_train_path)
            dataset_test_data, dataset_test_targets = load_preprocessed_dataset(save_test_path)

        else:
            # Preprocess and save
            dataset_data, dataset_targets = preprocess_dataset_from_imagefolder(
                train, save_path=save_train_path
            )
            dataset_test_data, dataset_test_targets = preprocess_dataset_from_imagefolder(
                test, save_path=save_test_path
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
    


def save_params_and_matrix(filename, params_dict, sparse_matrix):
    with open(filename + '.json', 'w') as f:
        json.dump(params_dict, f)
    sparse.save_npz(filename + '.npz', sparse_matrix)



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
        torch.cuda.empty_cache()


DEVICE = None
logger = None

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    ############################## Argument Parser ##############################
    parser = argparse.ArgumentParser(description="Run influence estimation with label noise")
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='ShallowMNIST', help='Model for experiment')
    parser.add_argument('--root_folder', type=str, default='../data', help='Root folder for data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for loaders')
    parser.add_argument('--img_size', type=int, default=32, help='Size to resize input image to')
    parser.add_argument('--config', type=str, default='none', choices=['pretrained_VGG16', 'pretrained_Resnet50', 'none'],
                        help='Learning config to use')
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
    num_workers  = args.num_workers
    img_size     = args.img_size
    config       = args.config

    # -------------- Device Setup --------------
    DEVICE = torch.device(args.device)
    if not torch.cuda.is_available():
        logger.log("WARN: Cuda unavailable, defaulting to cpu...", level=0)
        DEVICE = torch.device('cpu')
    logger.log("Using device: {}".format(DEVICE), level=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prerequisites()

    logger.log("Running experiment using dataset:{}, with model_name: {}...".format(dataset, model_name), level=1)

    loader_params = {
        'dataset_name':        dataset,
        'conversion':          'none',
        'root_folder':         root_folder,
        'training_size':       'full', # 'full'
        'batch_size':          400,
        'IG_batch_size':       1000, 
        'transform':           None,
        'add_singleton':       False,
        'convert_to_torch':    False,
        'num_workers':         num_workers,
    }
    train_params = {
        'optimizer':           'Adam',
        'scheduler':           {'name': None}, # 'step_size': 10, 'milestones':[10,20,30],'gamma':0.8, 'max_lr': 0.01}
        'init_rate':           0.0005,
        'total_epochs':        150,
        'weight_decay':        0, 
        'criterion':           'CrossEntropyLoss',
        'disp_epoch':          False,
        'disp_loss_epoch':     True,
        'disp_time_per_epoch': True,
        'disp_loss_final':     True, 
        'disp_accuracy_final': True
    }
    model_params = {
        'type':                ResNet if 'ResNet' in model_name else CNN,
        'name':                model_name,
        'in_channels':         1,
        'num_classes':         10,
        'img_size':            img_size,
        'batchnorm':           True,
        'fine_tune':           'NEW_LAYERS',
        'snapshot_k':          15,
    }

    # -------------- Customize arguments based on dataset --------------
    if dataset == 'MNIST' or dataset == 'FashionMNIST':
        model_params['in_channels'] = 1
        model_params['num_classes'] = 10
    
    elif dataset == 'CIFAR10':
        model_params['in_channels'] = 3
        model_params['num_classes'] = 10
    
    elif dataset == 'Flowers102':
        model_params['in_channels'] = 3
        model_params['num_classes'] = 102

    elif dataset == 'FGVCAircraft':
        model_params['in_channels'] = 3
        model_params['num_classes'] = 100


    model = get_model_from_params(model_params)

    image_size = (img_size, img_size)
    trainloader, testloader, IG_trainloader = genloaders_vision(
        loader_params,
        image_size=image_size,
        logger=logger
    )

    logger.log("Dataloaders generated, starting training snapshot...", level=1)

    model = model.to(device)
    model = model.train()

    # Snapshot dir to save data to
    snapshot_dir = dataset + '_snapshots'
    snapshot_k   = model_params['snapshot_k']
    
    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)

    all_train_losses = []
    for epoch in range(train_params['total_epochs']):
        
        if snapshot_dir is not None and (epoch) % snapshot_k == 0:
            snapshot_path = os.path.join(snapshot_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), snapshot_path)
            logger.log(f"Snapshot saved: {snapshot_path}", level=1)

        optimizer, scheduler, criterion = get_learning_config(model, train_params, config=config)

        if train_params['disp_epoch'] == True and logger is not None:
            logger.log("Running epoch {}...".format(epoch), level=1)
        
        train_loss = []
        loss_weights = []

        for inputs, labels, indices in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device):
                allouts = model(inputs)
                loss = criterion(allouts, labels.long())
    
            loss.backward()
            train_loss.append(loss.item())
            loss_weights.append(len(labels))
            optimizer.step()
            
        scheduler.step()
        all_train_losses.append(np.average(np.array(train_loss), weights=np.array(loss_weights)))
        
        if train_params['disp_loss_epoch'] == True and logger is not None:
            logger.log("Training Loss: {}".format(all_train_losses[-1]), level=1)
            logger.log("Accuracy: {}...".format(test_model(model, testloader)), level=1)
            model = model.train()

    if train_params['disp_accuracy_final'] == True and logger is not None:
        logger.log("Accuracy: {}...".format(test_model(model, testloader)), level=1)

    model = model.eval()

    params_bundle = {
        'model_params': model_params,
        'train_params': train_params,
        'loader_params': loader_params
    }
    # Save to pickle
    with open(os.path.join(snapshot_dir, 'config_bundle.pkl'), 'wb') as handle:
        pickle.dump(params_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.log("Saved config_bundle.pkl inside: {}".format(snapshot_dir), level=1)

        
    # -------------- Stop logger listener cleanly, and cleanup --------------
    logger.log("End experiment using dataset:{}, with model_name: {}...".format(dataset, model_name), level=1)
    logger.log("Shutting down logger listener...", level=1)
    log_queue.put(None)  # Sentinel to signal listener shutdown
    listener.join()

    gc.collect()
    torch.cuda.empty_cache()
                

        
