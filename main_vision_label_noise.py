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
from torch.utils import data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.functional as F
import numpy as np
import sys, os
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import pickle
import random
from PIL import Image
import scipy
import time 
import gc

import json


import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

from lib_train import *
from lib_cnn import * 
from lib_IGviz import *
from lib_influence_groundtruth import * 



def add_label_noise(targets, noise_type, noise_level, num_classes):
    np_targets = targets.numpy()
    num_noisy = int(noise_level * len(np_targets))
    noisy_indices = np.random.choice(len(np_targets), num_noisy, replace=False)

    if noise_type == 'symmetric':
        # Symmetric noise: randomly assign any class
        new_labels = np.random.choice(num_classes, num_noisy)
    elif noise_type == 'asymmetric':
        # Asymmetric noise: shift labels to the next class
        new_labels = np_targets[noisy_indices].copy()
        for i in range(num_noisy):
            new_labels[i] = (np_targets[noisy_indices[i]] + 1) % num_classes

    np_targets[noisy_indices] = new_labels
    return torch.from_numpy(np_targets)



def genloaders_vision(loader_params, labelnoise_params):

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
    
    transform_basic = transforms.ToTensor()
    transform_train = transform_test = transform_basic

    dataset_name = loader_params['dataset_name']
    root = loader_params['root_folder']
    
    ############################## Load relevant datasets ##############################
    
    if dataset_name == 'FashionMNIST':
        train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform_train)
        test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
        is_grayscale = True
    
    elif dataset_name == 'MNIST':
        train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform_train)
        test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        is_grayscale = True

    elif dataset_name == 'CIFAR10':
        train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        is_grayscale = False

    elif dataset_name == 'CIFAR100':
        train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        is_grayscale = False

    ############################## Additional fine grained datasets ##############################
    
    elif dataset_name == 'Flowers102':
        train = torchvision.datasets.Flowers102(root=root, split='train', download=True, transform=transform_train)
        test = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=transform_test)
        is_grayscale = False

    elif dataset_name == 'FGVCAircraft':
        train = torchvision.datasets.FGVCAircraft(root=root, split='train', download=True, transform=transform_train)
        test = torchvision.datasets.FGVCAircraft(root='./data', split='test', download=True, transform=transform_test)
        is_grayscale = False

    elif dataset_name == 'SVHN':
        train = torchvision.datasets.SVHN(root=root, split='train', download=True, transform=transform_train)
        test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        is_grayscale = False

    else:
        print("ERROR: Unknown dataset:", dataset_name)

    dataset_data, dataset_targets = preprocess_dataset(train, is_grayscale=is_grayscale)
    dataset_test_data, dataset_test_targets = preprocess_dataset(test, is_grayscale=is_grayscale)
    
    ############################## Apply Label Noise ##############################
    if labelnoise_params['noise_type'] is not None and labelnoise_params['noise_level'] > 0.0:
        num_classes = len(torch.unique(dataset_targets))
        dataset_targets = add_label_noise(
            dataset_targets,
            labelnoise_params['noise_type'], 
            labelnoise_params['noise_level'], num_classes
        )

    ############################## Generate DataLoaders ##############################
    trainloader, testloader, IG_trainloader = genloaders(
        dataset_data.to(DEVICE),
        dataset_targets.to(DEVICE),
        dataset_test_data.to(DEVICE),
        dataset_test_targets.to(DEVICE),
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
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.empty_cache()


DEVICE = None

if __name__ == "__main__":
    
    ############################## Argument Parser ##############################
    parser = argparse.ArgumentParser(description="Run influence estimation with label noise")
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='ShallowMNIST', help='Model for experiment')
    parser.add_argument('--root_folder', type=str, default='../data', help='Root folder for data')
    parser.add_argument('--noise_type', type=str, default='symmetric', choices=['symmetric', 'asymmetric', 'none'], help='Type of label noise')
    parser.add_argument('--program_mode', type=str, default='normal', choices=['normal', 'GT'], help='Run mode')
    parser.add_argument('--save_mode', type=str, default='load', choices=['load', 'store', 'none'], help='Save or load influence graph')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()

    # -------------- Device Setup --------------
    DEVICE = torch.device(args.device)
    if not torch.cuda.is_available():
        print("WARN: Cuda unavailable, defaulting to cpu...")
    print(f"Using device: {DEVICE}")

    prerequisites()

    # -------------- Unpack parser arguments --------------
    dataset      = args.dataset
    model_name   = args.model_name
    root_folder  = args.root_folder
    program_mode = args.program_mode # normal or GT (Ground truth)
    save_mode    = args.save_mode # store, load or none
    noise_types  = [args.noise_type] 
    noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for noise_type in noise_types:
        for noise_level in noise_levels:
            gc.collect()
            torch.cuda.empty_cache()
            
            labelnoise_params = {
                'noise_type':  noise_type,
                'noise_level': noise_level
            }
            
            loader_params = {
                'dataset_name':     dataset,
                'conversion':       'none',
                'root_folder':      root_folder,
                'training_size':    5000, # 'full'
                'batch_size':       40,
                'IG_batch_size':    400, 
                'transform':        None,
                'add_singleton':    False,
                'convert_to_torch': False,
                }
            
            influence_params = {
                'loss_scaling_span':  'full', # 'batch' or 'full'
                'loss_scaling_type':  'root_mean_squared', # 'mean' or 'mean_absolute' or None
                'set_zero_mean':      False,# 'full' or 'separate'
                'class_normalize' :   False, 
                'remove_negatives' :  False, 
                'clipping' :          False,
                'intraclass_only' :   True,
                'negative_clipping':  False,
                'clip_outliers':      False,
                'mode':               'mean', # For InfluenceGraphv3
                'gradient_lr':        0.1, # For InfluenceGraphv3
                # 'dtype':              np.float16,
                'dtype':              np.float32,
                'graph_type':         InfluenceGraphv4,
                }
            
            train_params = {
                'optimizer':           'Adam',
                'scheduler':           {'name': None}, # 'step_size': 10, 'milestones':[10,20,30],'gamma':0.8, 'max_lr': 0.01}
                'init_rate':           0.0005,
                'total_epochs':        100,
                'weight_decay':        0, 
                'criterion':           'CrossEntropyLoss',
                'disp_epoch':          False,
                'disp_loss_epoch':     False,
                'disp_time_per_epoch': True, 
                'disp_loss_final':     True, 
                'disp_accuracy_final': True
                }
            
            influence_GT_params = {
                'type':                'batch', # batch or representative
                'training_iterations': train_params['total_epochs'],
                'intraclass_only':     True,
                # 'dtype':               np.float16
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
                'type':        CNN,
                'name':        model_name,
                'in_channels': 1,
                'batchnorm':   True,
                }
            
            model = model_params['type'](model_params['name'], model_params['in_channels'], batchnorm = model_params['batchnorm'])
        
            trainloader, testloader, IG_trainloader = genloaders_vision(loader_params, labelnoise_params)
            
            if program_mode == 'GT':
                if save_mode == 'load':
                    node_size = trainloader.dataset.inputs.shape[0]            
                    IG_GT = InfluenceGraph_GT(
                        node_size,
                        trainloader.dataset.labels.squeeze().cpu().numpy(),
                        trainloader.batch_size,influence_GT_params
                    )
                    graphmat = IG_GT.load_graph('IG-DB', loader_params['dataset_name'], 'latest')
                else:            
                    IG_GT = batch_influence_GT(
                        model_params,
                        trainloader,
                        IG_trainloader,
                        influence_GT_params,
                        influence_GT_train_params,
                        loader_params
                    )
                    IG_GT.store_graph('IG-DB', loader_params, influence_GT_params, influence_GT_train_params)
                    graphmat = IG_GT.normgraph_mat
                
            elif program_mode == 'normal':
                if save_mode == 'load':
                    model_IG = influence_params['graph_type'](
                        trainloader.dataset.inputs.shape[0],
                        trainloader.dataset.labels.squeeze().cpu().numpy(),
                        loader_params['batch_size'],
                        influence_params
                    )
                    graphmat = model_IG.load_graph('IG-DB', loader_params['dataset_name'],'latest')
                else:
                    model, all_train_losses, model_IG = estimate_influencegraph(
                        model,
                        trainloader,
                        IG_trainloader,
                        train_params,
                        influence_params,
                        loader_params
                    )
                    test_accuracy = test_model(model, testloader)
        
                    model_IG.update_normalized_graph()
                    graphmat = model_IG.normgraph_mat
                    
                    mean_in_degree = np.mean(graphmat.max(axis=0))
                    if save_mode == 'store':
                        new_model_params = {
                        k: v.__name__ if isinstance(v, type) else v for k, v in model_params.items()
                    }
                        new_influence_params = {
                        k: v.__name__ if isinstance(v, type) else v for k, v in influence_params.items()
                    }
                        params_dict = {
                            'labelnoise_params': labelnoise_params,
                            'loader_params': loader_params,
                            'influence_params': new_influence_params,
                            'train_params': train_params,
                            'model_params': new_model_params,
                            'test_accuracy': test_accuracy,
                            'mean_in_degree': mean_in_degree, 
                        }
                        print('Test Accuracy:', test_accuracy)
                        print('Mean In Degree:', mean_in_degree)
                        
                        folder_name = os.path.join('Label noise', loader_params['dataset_name']+' (alpha values zero mean 5k)')
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                        
                        # Modify the filename
                        filename = os.path.join(folder_name, labelnoise_params['noise_type'] + str(labelnoise_params['noise_level']))
                        save_params_and_matrix(filename, params_dict, graphmat)
                            
                        # sparse.save_npz(loader_params['dataset_name']+'_'+labelnoise_params['noise_type']+
                        #                 labelnoise_params['noise_level']+".npz", self.normgraph_mat)
        
                
            # 
            # ''
            # model_IG.store_graph('IG-DB', loader_params, influence_params, train_params)
            #
            vis_influencepairs(graphmat, trainloader.dataset.inputs, max_percentile = 1, num_pairs=25)
            print('checkpoint')
            vis_influencepairs(graphmat, trainloader.dataset.inputs, min_percentile = 99, num_pairs=25)

            vis_influencenodes(graphmat, trainloader.dataset.inputs, max_percentile = 3, num_nodes = 25)
            print('checkpoint')
            vis_influencenodes(graphmat, trainloader.dataset.inputs, min_percentile = 97, num_nodes = 25) 
            # 
            print('done')
                

        