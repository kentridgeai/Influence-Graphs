# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 15:31:14 2025

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


import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

from lib_train import *
from lib_cnn import * 
from lib_IGviz import *
from lib_influence_groundtruth import * 
from lib_graphops import * 

from sklearn.cluster import KMeans


def core_set_pruning(data, prune_size):
    """
    Perform Core-Set Selection using K-Means clustering.
    Args:
        data: Dataset features (e.g., flattened images or embeddings).
        prune_size: Number of samples to retain.
    Returns:
        Selected indices for the pruned dataset.
    """
    kmeans = KMeans(n_clusters=prune_size, random_state=42)
    kmeans.fit(data)
    cluster_centers = kmeans.cluster_centers_
    selected_indices = []

    for center in cluster_centers:
        # Find the closest data point to each cluster center
        distances = np.linalg.norm(data - center, axis=1)
        selected_indices.append(np.argmin(distances))
    
    return selected_indices



def directed_spectral_clustering(graphmat, n_clusters):
    """
    Perform Directed Spectral Clustering on the influence graph.
    Args:
        graphmat: Influence graph adjacency matrix (NxN).
        n_clusters: Number of clusters (desired dataset size after pruning).
    Returns:
        pruned_indices: Indices of representative nodes for each cluster.
    """
    # Ensure the graph matrix is dense
    if hasattr(graphmat, "toarray"):
        graphmat = graphmat.toarray()

    # Step 1: Compute row and column sums
    row_sum = np.sum(graphmat, axis=1)  # Out-degrees
    col_sum = np.sum(graphmat, axis=0)  # In-degrees

    # Step 2: Construct the asymmetric normalized Laplacian
    D_row_inv = np.diag(1.0 / (row_sum + 1e-10))  # Add epsilon to avoid division by zero
    L = np.dot(D_row_inv, graphmat)  # Transition matrix

    # Step 3: Perform spectral embedding (eigen decomposition)
    # Use the top k eigenvectors corresponding to the largest eigenvalues
    eigvals, eigvecs = np.linalg.eig(L)  # Compute eigenvalues and eigenvectors
    top_k_indices = np.argsort(eigvals)[::-1][:n_clusters]  # Get top-k eigenvalues
    spectral_embedding = eigvecs[:, top_k_indices]  # Top-k eigenvectors (NxK)

    # Step 4: Cluster the nodes using K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(spectral_embedding)

    # Step 5: Find representative nodes for each cluster
    pruned_indices = []
    for cluster_id in range(n_clusters):
        cluster_nodes = np.where(labels == cluster_id)[0]
        cluster_matrix = graphmat[cluster_nodes][:, cluster_nodes]
        centroid = cluster_matrix.mean(axis=0)
        distances = np.linalg.norm(cluster_matrix - centroid, axis=1)
        closest_node = cluster_nodes[np.argmin(distances)]
        pruned_indices.append(closest_node)

    return pruned_indices



def genloaders_vision(loader_params):

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

    ############################## Generate DataLoaders ##############################
    trainloader, testloader, IG_trainloader = genloaders(
        dataset_data.to(DEVICE),
        dataset_targets.to(DEVICE),
        dataset_test_data.to(DEVICE),
        dataset_test_targets.to(DEVICE),
        loader_params
    )
        
    return trainloader, testloader, IG_trainloader
    


def gen_prunedloaders_vision(loader_params):

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

    ############################## Generate DataLoaders ##############################
    trainloader, testloader = gen_pruned_loaders(
        dataset_data.to(DEVICE),
        dataset_targets.to(DEVICE),
        dataset_test_data.to(DEVICE),
        dataset_test_targets.to(DEVICE),
        loader_params
    )
    
    return trainloader, testloader



def get_accuracy(loader_params, train_params):
    
    trainloader, testloader = gen_prunedloaders_vision(loader_params)
    
    model_temp = model_params['type'](model_params['name'], model_params['in_channels'], batchnorm = model_params['batchnorm'])
    model_temp, all_train_losses = train_model_general(model_temp, trainloader, train_params)
    accuracy_model_temp = test_model(model_temp, testloader)
    
    return accuracy_model_temp



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

    prune_size = 500
    prune_ratios = np.linspace(1, 0.4, 10)
    
    loader_params = {
        'dataset_name':     dataset,
        'conversion':       'none',
        'root_folder':      root_folder,
        'training_size':    1000, # 'full'
        'batch_size':       20,
        'IG_batch_size':    400,
        'transform':        None,
        'add_singleton':    False,
        'convert_to_torch': False,
    }
    
    loader_params_final = {
        'dataset_name':     dataset,
        'conversion':       'none',
        'root_folder':      root_folder,
        'training_size':    1000,
        'train_indices':    'all',# 'full'
        'batch_size':       400, 
        'transform':        None,
        'add_singleton':    False,
        'convert_to_torch': False,
    }
    
    influence_params = {
        'loss_scaling_span': 'full', # 'batch' or 'full'
        'loss_scaling_type': 'root_mean_squared', # 'mean' or 'mean_absolute' or None
        'set_zero_mean':     False,
        'class_normalize' :  False, 
        'remove_negatives' : False, 
        'clipping' :         False,
        'intraclass_only' :  True,
        'negative_clipping': False,
        'clip_outliers':     True, 
        'mode':              'mean', # For InfluenceGraphv3
        'gradient_lr':       0.1, # For InfluenceGraphv3
        # 'dtype':             np.float16,
        'dtype':             np.float32, 
        'graph_type':        InfluenceGraphv4,
    }
    
    train_params = {
        'optimizer':           'Adam',
        'scheduler':           {'name': None}, # 'step_size': 10, 'milestones':[10,20,30],'gamma':0.8, 'max_lr': 0.01}
        'init_rate':           0.001,
        'total_epochs':        100,
        'weight_decay':        0, 
        'criterion':           'CrossEntropyLoss',
        'disp_epoch':          False,
        'disp_loss_epoch':     False,
        'disp_time_per_epoch': True, 
        'disp_loss_final':     False, 
        'disp_accuracy_final': True
    }
    
    influence_GT_params = {
        'type':                'batch', # batch or representative
        'training_iterations': train_params['total_epochs'],
        'intraclass_only':     True,
        # 'dtype': np.float16
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
        'batchnorm':           True,
    }
    
    model = model_params['type'](model_params['name'], model_params['in_channels'], batchnorm = model_params['batchnorm'])

    trainloader, testloader, IG_trainloader,  = genloaders_vision(loader_params)
    
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
        
    
    train_params['total_epochs'] = 400
    # train_params['total_epochs'] = 400
    
    accuracy_main_model = get_accuracy(loader_params_final,train_params)
    
    
    print("Accuracy of Native Model:", accuracy_main_model)
    # print("Accuracy of Model with max-weight IG pruning:", accuracy_max_model_pruned)
    # print("Accuracy of Model with mean-weight IG pruning:", accuracy_mean_model_pruned)
    # print("Accuracy of Model with random pruning of training dataset:", accuracy_random_model_pruned)
    
    # acc_max = []
    acc_random = []
    acc_iterative = [] 
    acc_coverage = [] 
    for r in prune_ratios:
        prune_size = int(r * loader_params['training_size'])
    
        # Initialize accumulators for averaging accuracies
        acc_random_runs = []
        acc_iterative_runs = []
        acc_coverage_runs = []  # For your IG_coverageprune_data approach
        
        # data = trainloader.dataset.inputs.cpu().numpy()
        # n_samples = data.shape[0]  # Total number of samples
        # if len(data.shape) > 2:
        #     data = data.reshape(n_samples, -1)# Assuming inputs are in NumPy format
        # selected_indices = core_set_pruning(data, prune_size)
        
        for _ in range(5):  # Train each network 4 times
            # Random Pruning
            train_indices = random.sample(range(loader_params['training_size']), prune_size)
            loader_params_final['train_indices'] = train_indices
            accuracy_random_model_pruned = get_accuracy(loader_params_final, train_params)
            acc_random_runs.append(accuracy_random_model_pruned)
            
            if r == 1.0:
                print('hajibu')
                acc_coverage_runs.append(accuracy_random_model_pruned)
                acc_iterative_runs.append(accuracy_random_model_pruned)
            else:
            # Core-Set Pruning
            
            # loader_params_final['train_indices'] = selected_indices
            # accuracy_core_set_model_pruned = get_accuracy(loader_params_final, train_params)
            # acc_core_set_runs.append(accuracy_core_set_model_pruned)
    
            # Your Approach: IG Coverage Pruning
            # train_indices, graphmat_pruned = IG_clusterprune_data(graphmat, target_num_nodes=prune_size)
            # train_indices,graphmat_pruned = IG_iterativecoverageprune_data(graphmat, target_num_nodes=prune_size, max_iterations=500)
                train_indices, graphmat_pruned = IG_adversaryprune_data(graphmat, target_num_nodes=prune_size, batch_remove=1, mode='mean')
                print(len(train_indices))
                # print(train_indices)
                loader_params_final['train_indices'] = train_indices
                accuracy_mean_model_pruned = get_accuracy(loader_params_final, train_params)
                acc_coverage_runs.append(accuracy_mean_model_pruned)
                
                
                train_indices,graphmat_pruned = IG_adversaryprune_data(graphmat, target_num_nodes=prune_size, batch_remove=1, mode='max')
                print(len(train_indices))
                # print(train_indices)
                loader_params_final['train_indices'] = train_indices
                accuracy_iterative_model_pruned = get_accuracy(loader_params_final, train_params)
                acc_iterative_runs.append(accuracy_iterative_model_pruned)
            
                print("Iter:", accuracy_iterative_model_pruned)
                print("Direct:", accuracy_mean_model_pruned)
                print("Random:", accuracy_random_model_pruned)
                
            
            
            
    
        # Calculate the average accuracy for this pruning ratio
        acc_random.append(np.mean(acc_random_runs))
        acc_iterative.append(np.mean(acc_iterative_runs))
        acc_coverage.append(np.mean(acc_coverage_runs))

# Plot the results

    # print(acc_core_set)
    print(acc_coverage)
    plt.plot(prune_ratios, acc_random, label="Random Pruning", linestyle='-.', marker='^')
    plt.plot(prune_ratios, acc_iterative, label="Adversary Pruning (Max)", linestyle='--', marker='s')
    plt.plot(prune_ratios, acc_coverage, label="Adversary Pruning (mean)", linestyle='-', marker='o')
    
    # Add labels, legend, and title
    plt.xlabel("Dataset Pruning Ratio")
    plt.ylabel("Average Accuracy")
    plt.title("Dataset Pruning Accuracy Comparison (Averaged over 4 Runs)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()


    
    
    

    # vis_influencepairs(graphmat,trainloader.dataset.inputs, max_percentile = 1, num_pairs=25)
    # print('checkpoint')
    # vis_influencepairs(graphmat,trainloader.dataset.inputs, min_percentile = 99, num_pairs=25)
    
    
    
    # vis_influencenodes(graphmat,trainloader.dataset.inputs, max_percentile = 3, num_nodes = 25)
    # print('checkpoint')

    # vis_influencenodes(graphmat,trainloader.dataset.inputs, min_percentile = 97, num_nodes = 25)
    
    # 

    print('done')
    

    