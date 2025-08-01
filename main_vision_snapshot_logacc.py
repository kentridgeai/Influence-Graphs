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
    
    transform_train = transforms.Compose(
        [
          # torchvision.transforms.GaussianBlur(5, sigma=2.0),
          # torchvision.transforms.functional.rgb_to_grayscale
         transforms.ToTensor(),
         ])
    transform_test = transforms.Compose(
        [
            # torchvision.transforms.GaussianBlur(5, sigma=2.0),
         transforms.ToTensor(),
         ])



    
    if loader_params['dataset_name'] == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST(root=loader_params['root_folder'], train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform_test)
        dataset.data = dataset.data.float()/255.0
        dataset_test.data = dataset_test.data.float()/255.0        
        
        dataset.data = dataset.data.unsqueeze(1)
        dataset_test.data = dataset_test.data.unsqueeze(1)
    
    if loader_params['dataset_name'] == 'MNIST':
        dataset = torchvision.datasets.MNIST(root=loader_params['root_folder'], train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.MNIST(root=loader_params['root_folder'], train=False,
                                                download=True, transform=transform_test)
        dataset.data = dataset.data.float()/255.0
        dataset_test.data = dataset_test.data.float()/255.0        
        
        dataset.data = dataset.data.unsqueeze(1)
        dataset_test.data = dataset_test.data.unsqueeze(1)
    
    elif loader_params['dataset_name'] == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root=loader_params['root_folder'], train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.CIFAR10(root=loader_params['root_folder'], train=False,
                                                download=True, transform=transform_test)
        dataset.data = torch.from_numpy(dataset.data)
        dataset_test.data = torch.from_numpy(dataset_test.data)
        
        dataset.data = dataset.data.float()/255.0
        dataset_test.data = dataset_test.data.float()/255.0        
        
        dataset.data = torch.permute(dataset.data,(0,3,1,2))
        dataset.targets = torch.from_numpy(np.array(dataset.targets))
        dataset_test.data = torch.permute(dataset_test.data,(0,3,1,2))
        dataset_test.targets = torch.from_numpy(np.array(dataset_test.targets))
        
    elif loader_params['dataset_name'] == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root=loader_params['root_folder'], train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.CIFAR100(root=loader_params['root_folder'], train=False,
                                                download=True, transform=transform_test)
        dataset.data = torch.permute(torch.from_numpy(dataset.data),(0,3,1,2))
        dataset.targets = torch.from_numpy(np.array(dataset.targets))
        dataset.data = dataset.data.float()/255.0
        dataset_test.data = torch.permute(torch.from_numpy(dataset_test.data),(0,3,1,2))
        dataset_test.targets = torch.from_numpy(np.array(dataset_test.targets))
        dataset_test.data = dataset_test.data.float()/255.0
        
    if labelnoise_params['noise_type'] is not None and labelnoise_params['noise_level']  > 0.0:
        num_classes = len(torch.unique(dataset.targets))
        dataset.targets = add_label_noise(dataset.targets, labelnoise_params['noise_type'], 
                                          labelnoise_params['noise_level'], num_classes)


        
    print(dataset.data.shape)
    trainloader, testloader, IG_trainloader = genloaders(dataset.data , dataset.targets, 
                                                         dataset_test.data, dataset_test.targets, loader_params)
        
    return trainloader, testloader, IG_trainloader

# +
def prerequisites():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    gc.collect()
    torch.cuda.empty_cache()
# -



def save_params_and_matrix(filename, params_dict, sparse_matrix):
    with open(filename + '.json', 'w') as f:
        json.dump(params_dict, f)
    sparse.save_npz(filename + '.npz', sparse_matrix)


import re


# +
if __name__ == "__main__":
    
    prerequisites() 
    program_mode = 'normal' # normal or GT (Ground truth)
    save_mode = 'store' # store, load or none


    print(torch.cuda.device_count())

    gc.collect()
    torch.cuda.empty_cache()


#     labelnoise_params = {
#         'noise_type': 'None', # symmetric, asymmetric or None
#         'noise_level': 0

#         }
#     loader_params = {
#         'dataset_name': 'MNIST',
#         'conversion': 'none',
#         'root_folder': '../data',
#         'training_size': 5000, # 'full'
#         'batch_size': 20,
#         'IG_batch_size': 1000, 
#         'transform': None,
#         'add_singleton': False,
#         'convert_to_torch': False,
#         }

    influence_params = {
        'loss_scaling_span':  'full', # 'batch' or 'full'
        'loss_scaling_type':  'root_mean_squared', # 'mean' or 'mean_absolute' or None
        'set_zero_mean': False,# 'full' or 'separate'
        'class_normalize' : False, 
        'remove_negatives' : False, 
        'clipping' : False,
        'intraclass_only' : True,
        'negative_clipping': False,
        'mode': 'mean', # For InfluenceGraphv3
        'gradient_lr': 0.1, # For InfluenceGraphv3
        'dtype': np.float16, 
        'graph_type': InfluenceGraphv5,
        }
    
    log_queue = Queue()
    logger = InfluenceLogger(log_queue, verbose=0)
    logger.log("Main process started.", level=1)

#     train_params = {
#         'optimizer': 'Adam',
#         'scheduler': {'name': None}, # 'step_size': 10, 'milestones':[10,20,30],'gamma':0.8, 'max_lr': 0.01}
#         'init_rate': 0.0005,
#         'total_epochs': 10,
#         'weight_decay': 0, 
#         'criterion': 'CrossEntropyLoss',
#         'disp_epoch': False,
#         'disp_loss_epoch': False,
#         'disp_time_per_epoch': True, 
#         'disp_loss_final': True, 
#         'disp_accuracy_final': True
#         }

    influence_GT_params={
        'type': 'batch', # batch or representative
        'training_iterations': 10,
        'intraclass_only': True,
        'dtype': np.float16
        }
    influence_GT_train_params={
        'optimizer': 'SGD',
        'scheduler': {'name': None}, # 'step_size': 10, 'milestones':[10,20,30],'gamma':0.8, 'max_lr': 0.01}
        'init_rate': 0.1,
        'total_epochs': 30,
        'weight_decay': 0, 
        'criterion': 'CrossEntropyLoss',
        'disp_epoch': False,
        'disp_loss_epoch': True,
        'disp_time_per_batch': True,
        'disp_total_time': True,
        }

#     model_params={
#         'type': CNN,
#         'name': 'ShallowMNIST',
#         'in_channels': 1,
#         'batchnorm': True,
#         }

    

    # Build folder name
    folder_name = "MNIST_lowerlr"
    
    config_path = os.path.join(folder_name, "config_bundle.pkl")

    with open(config_path, 'rb') as handle:
        params_bundle = pickle.load(handle)

    # Unpack the dicts
    model_params        = params_bundle['model_params']
    train_params        = params_bundle['train_params']
    loader_params       = params_bundle['loader_params']
    labelnoise_params   = params_bundle['labelnoise_params']

    model = model_params['type'](model_params['name'], model_params['in_channels'], batchnorm = model_params['batchnorm'])

    loader_params['training_size'] = 5000
    loader_params['batch_size'] = 10
    trainloader, testloader, IG_trainloader = genloaders_vision(loader_params, labelnoise_params)

    
    model_files = sorted([
        f for f in os.listdir(folder_name)
        if f.endswith('.pt')
    ])
        

    # Load each snapshot into the existing model instance

# Load each snapshot into the existing model instance
    train_acc_log = os.path.join(folder_name, "train_accuracy.txt")
    test_acc_log = os.path.join(folder_name, "test_accuracy.txt")
    
    # Clear previous logs
    open(train_acc_log, 'w').close()
    open(test_acc_log, 'w').close()
    
    # Helper function to extract epoch number
    def get_epoch(fname):
        match = re.search(r"model_epoch_(\d+)\.pt", fname)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Invalid filename format: {fname}")
    
    # Sort model files by epoch number
    sorted_model_files = sorted(model_files, key=get_epoch)
    
    for fname in sorted_model_files:
        snapshot_path = os.path.join(folder_name, fname)
        model.load_state_dict(torch.load(snapshot_path))
        print(f"Loaded: {fname}")
    
        epoch_tag = f"epoch_{get_epoch(fname)}"
    
        # Compute accuracies
        train_acc = test_model(model, trainloader)
        test_acc = test_model(model, testloader)
    
        # Append to log files
        with open(train_acc_log, 'a') as f_train:
            f_train.write(f"{epoch_tag}: {train_acc:.4f}\n")
    
        with open(test_acc_log, 'a') as f_test:
            f_test.write(f"{epoch_tag}: {test_acc:.4f}\n")

        # IG_GT = batch_influence_GT_from_model(model, trainloader, IG_trainloader, influence_GT_params, influence_GT_train_params, loader_params,
        #                                      logger = logger)
        # graphmat = IG_GT.normgraph_mat
# 
        # npz_path = os.path.join(folder_name, f"{epoch_tag}.npz")
        # sparse.save_npz(npz_path, graphmat)

        # ➤ Save metadata
        # pkl_path = os.path.join(folder_name, f"{epoch_tag}.pkl")
        # with open(pkl_path, 'wb') as handle:
        #     pickle.dump([
        #         loader_params,
        #         influence_params,
        #         train_params,
        #         IG_GT.node_labels,
        #         IG_GT.transform_params
        #     ], handle, protocol=pickle.HIGHEST_PROTOCOL)

        # print(f"Saved: {epoch_tag}.npz and {epoch_tag}.pkl")



# -



