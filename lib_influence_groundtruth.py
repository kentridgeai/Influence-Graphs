# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 20:33:44 2025

@author: User
"""

import os
import numpy as np
import itertools
import pickle
import time
import torch.nn as nn
import torch.optim as optim

from scipy import sparse
from scipy.sparse import csr_matrix
from torch import autocast
from torch.cuda.amp import GradScaler

from lib_cnn import *
from lib_train import *


dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
    


def update_IG_GT(IG, main_model, batch_indices, old_trainloss, IG_trainloader, train_params, loader_params, logger=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'
    
    old_trainloss = copy.deepcopy(old_trainloss)
    model = copy.deepcopy(main_model)
    model = model.eval()
    
    batch_indices = batch_indices.cpu()
    old_batchloss = old_trainloss[batch_indices]
    
    if train_params['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')
        
    elif train_params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        
    elif train_params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss(reduction = 'none')

    labels_batch = IG_trainloader.dataset.labels[batch_indices]
    target_labels = torch.unique(labels_batch)
    all_labels = IG_trainloader.dataset.labels  # assumed to be list or tensor

    # Find matching indices where label ∈ target_labels
    mask             = torch.isin(all_labels, target_labels)
    matching_indices = torch.nonzero(mask).squeeze().tolist()
    
    # Build filtered loader
    subset = Subset(IG_trainloader.dataset, matching_indices)
    filtered_loader = DataLoader(
        subset,
        batch_size=loader_params['IG_batch_size'],
        pin_memory=True
    )

    trainloss = np.zeros(IG.node_size)

    # Compute losses
    with torch.no_grad():
        for inputs, labels, indices in filtered_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixed precision if on GPU
            if use_amp:
                with torch.autocast(device_type=device):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.long())
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                
            trainloss[indices.cpu()] = loss.cpu()

    batchloss_diff = old_batchloss - trainloss[batch_indices]
    trainloss_diff = old_trainloss - trainloss

    scale_ref      = copy.copy(trainloss_diff[matching_indices])
    scale_ref      = np.sqrt(np.mean(scale_ref**2))
    batchloss_diff = batchloss_diff / scale_ref
    trainloss_diff = trainloss_diff / scale_ref

    IG.update_influence_graph(
        batch_indices,
        batchloss_diff,
        trainloss_diff
    )
    return IG, np.mean(trainloss[batch_indices.cpu()])



def batch_influence_GT(model_params,
                       trainloader,
                       IG_trainloader,
                       influence_GT_params,
                       influence_GT_train_params,
                       loader_params,
                       config=None,
                       logger=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'
    
    node_size = trainloader.dataset.inputs.shape[0]
    
    IG_GT = InfluenceGraphv5(
        node_size,
        trainloader.dataset.labels.squeeze().cpu().numpy(),
        trainloader.batch_size,
        influence_GT_params
    )
        
    if influence_GT_train_params['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
        
    elif influence_GT_train_params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        
    elif influence_GT_train_params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss()

    trainloss = np.zeros(node_size)

    model = get_model_from_params(model_params)
    model = model.to(device)
    model = model.half()

    state_dict = copy.deepcopy(model.state_dict())
    
    with torch.no_grad():
        for inputs, labels, indices in IG_trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixed precision if on GPU
            if use_amp:
                with torch.autocast(device_type=device):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.long())
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())

            trainloss[indices.cpu()] = loss.cpu()

    logger.reset_timer()
    for epoch in range(influence_GT_params['training_iterations']):
        gc.collect()
        # logger.reset_timer()
        
        if logger is not None:
            logger.log("Starting batch influence_GT iteration: {}...".format(epoch), level=1)

        count = 0
        for inputs, labels, indices in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            if logger is not None and count % 20 == 0:
                logger.log("Reference time of 20 trainloader thingy : {}...", level=2)

            count += 1
                
            # model = get_model_from_params(model_params)
            model.load_state_dict(state_dict)

            optimizer, scheduler, criterion = get_learning_config(model, influence_GT_train_params, config=config)

            model = model.to(device)
            model = model.train()
            model = model.half()
            
            for mini_epoch in range(influence_GT_train_params['total_epochs']):
                
                if influence_GT_train_params['disp_epoch'] == True and logger is not None:
                    logger.log("Mini epoch influence_GT iteration: {}...".format(mini_epoch), level=2)
                
                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    # Mixed precision on GPU
                    with torch.autocast(device_type=device):
                        allouts = model(inputs)
                        loss = criterion(allouts, labels.long())
    
                    loss.backward()
                    optimizer.step()
                
                else:
                    # Standard FP32 on CPU
                    allouts = model(inputs)
                    loss = criterion(allouts, labels.long())
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
            IG_GT, mean_trainloss = update_IG_GT(
                IG_GT,
                model,
                indices,
                trainloss,
                IG_trainloader,
                influence_GT_train_params,
                loader_params,
                logger=logger
            )
                
            if influence_GT_train_params['disp_loss_epoch'] == True and logger is not None:
                logger.log("Training Loss: {}...".format(mean_trainloss), level=2)
            
    IG_GT.update_normalized_graph()
    
    return IG_GT


