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


import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

from lib_train import *
from lib_cnn import * 
from lib_IGviz import *
from lib_influence_groundtruth import * 



def genloaders_vision(loader_params):
    
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
        dataset_test = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform_test)
        dataset.data = dataset.data.float()/255.0
        dataset_test.data = dataset_test.data.float()/255.0        
        
        dataset.data = dataset.data.unsqueeze(1)
        dataset_test.data = dataset_test.data.unsqueeze(1)
    
    elif loader_params['dataset_name'] == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root=loader_params['root_folder'], train=True,
                                                download=True, transform=transform_train)
        dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False,
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
        dataset_test = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform_test)
        dataset.data = torch.permute(torch.from_numpy(dataset.data),(0,3,1,2))
        dataset.targets = torch.from_numpy(np.array(dataset.targets))
        dataset.data = dataset.data.float()/255.0
        dataset_test.data = torch.permute(torch.from_numpy(dataset_test.data),(0,3,1,2))
        dataset_test.targets = torch.from_numpy(np.array(dataset_test.targets))
        dataset_test.data = dataset_test.data.float()/255.0
        
        
    trainloader, testloader, IG_trainloader = genloaders(dataset.data.cuda(), dataset.targets.cuda(), 
                                                         dataset_test.data.cuda(), dataset_test.targets.cuda(), loader_params)
        
    return trainloader, testloader, IG_trainloader

def prerequisites():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    gc.collect()
    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    
    prerequisites() 
    program_mode = 'normal' # normal or GT (Ground truth)
    save_mode = 'none' # store, load or none
    
    loader_params = {
        'dataset_name': 'MNIST',
        'conversion': 'none',
        'root_folder': '../data',
        'training_size': 5000, # 'full'
        'batch_size': 20,
        'IG_batch_size': 400, 
        'transform': None,
        'add_singleton': False,
        'convert_to_torch': False,
        }
    
    influence_params = {
        'loss_scaling_span':  'full', # 'batch' or 'full'
        'loss_scaling_type':  'root_mean_squared', # 'mean' or 'mean_absolute' or None
        'set_zero_mean': False,
        'class_normalize' : False, 
        'remove_negatives' : False, 
        'clipping' : False,
        'intraclass_only' : True,
        'negative_clipping': False,
        'mode': 'mean', # For InfluenceGraphv3
        'gradient_lr': 0.1, # For InfluenceGraphv3
        'dtype': np.float16, 
        'graph_type': InfluenceGraphv4,
        }
    
    
    train_params = {
        'optimizer': 'Adam',
        'scheduler': {'name': None}, # 'step_size': 10, 'milestones':[10,20,30],'gamma':0.8, 'max_lr': 0.01}
        'init_rate': 0.0005,
        'total_epochs': 100,
        'weight_decay': 0, 
        'criterion': 'CrossEntropyLoss',
        'disp_epoch': False,
        'disp_loss_epoch': True,
        'disp_time_per_epoch': True, 
        'disp_loss_final': False, 
        'disp_accuracy_final': True
        }
    
    influence_GT_params={
        'type': 'batch', # batch or representative
        'training_iterations': train_params['total_epochs'],
        'intraclass_only': True,
        'dtype': np.float16
        }
    influence_GT_train_params={
        'optimizer': 'SGD',
        'scheduler': {'name': None}, # 'step_size': 10, 'milestones':[10,20,30],'gamma':0.8, 'max_lr': 0.01}
        'init_rate': 0.1,
        'total_epochs': 50,
        'weight_decay': 0, 
        'criterion': 'CrossEntropyLoss',
        'disp_epoch': False,
        'disp_loss_epoch': True,
        'disp_time_per_batch': True,
        'disp_total_time': True,
        
        }
    
    model_params={
        'type': CNN,
        'name': 'ShallowMNIST',
        'in_channels': 1,
        'batchnorm': True,
        }
    
    model = model_params['type'](model_params['name'], model_params['in_channels'], batchnorm = model_params['batchnorm'])

    trainloader, testloader, IG_trainloader = genloaders_vision(loader_params)
    
    if program_mode == 'GT':
        if save_mode == 'load':
            node_size = trainloader.dataset.inputs.shape[0]            
            IG_GT = InfluenceGraph_GT(node_size, trainloader.dataset.labels.squeeze().cpu().numpy(),
                              trainloader.batch_size,influence_GT_params)
            graphmat = IG_GT.load_graph('IG-DB', loader_params['dataset_name'], 'latest')
        else:            
            IG_GT = batch_influence_GT(model_params, trainloader, IG_trainloader, influence_GT_params, influence_GT_train_params,loader_params)
            IG_GT.store_graph('IG-DB', loader_params, influence_GT_params, influence_GT_train_params)
            graphmat = IG_GT.normgraph_mat
        
    elif program_mode == 'normal':
        if save_mode == 'load':
            model_IG = influence_params['graph_type'](trainloader.dataset.inputs.shape[0],trainloader.dataset.labels.squeeze().cpu().numpy(),
                                      loader_params['batch_size'],influence_params)
            graphmat = model_IG.load_graph('IG-DB', loader_params['dataset_name'],'latest')
        else:
            model,all_train_losses,model_IG = estimate_influencegraph(model,trainloader, IG_trainloader,
                                                                      train_params, influence_params,loader_params)
            
            model_IG.update_normalized_graph()
            graphmat = model_IG.normgraph_mat
            if save_mode == 'store':
                model_IG.store_graph('IG-DB', loader_params, influence_params, train_params)
                # sparse.save_npz('MNIST_max_absolute.npz', model_IG.normgraph_mat)
        
    
        
    # 
    
    # ''
    # model_IG.store_graph('IG-DB', loader_params, influence_params, train_params)
    
    
    # 
    

    vis_influencepairs(graphmat,trainloader.dataset.inputs, max_percentile = 1, num_pairs=25)
    print('checkpoint')
    vis_influencepairs(graphmat,trainloader.dataset.inputs, min_percentile = 99, num_pairs=25)
    
    
    
    vis_influencenodes(graphmat,trainloader.dataset.inputs, max_percentile = 3, num_nodes = 25)
    print('checkpoint')

    vis_influencenodes(graphmat,trainloader.dataset.inputs, min_percentile = 97, num_nodes = 25)
    
    # 

    print('done')
    

    