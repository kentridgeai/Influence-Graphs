# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:48:11 2025

@author: User
"""

import os
import matplotlib.pyplot as plt
import copy
import gc
import sys
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision

from collections import defaultdict
from datetime import datetime
from multiprocessing import Queue, Process, current_process
from torch.utils import data
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torch.cuda.amp import GradScaler
from torch import autocast

from lib_graph import * 
from lib_preprocessing import *


dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)



class Dataset_v2(data.Dataset):
    # Characterizes a dataset for PyTorch
    
    def __init__(self, inputs, labels, transform=None):
        # 'Initialization'
        self.labels = labels
        self.inputs = inputs
        self.transform = transform
        
    def __len__(self):
        # 'Denotes the total number of samples'
        return self.inputs.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        
        img = self.inputs[index]

        if self.transform is not None:
            img = self.transform(img)

        y = int(self.labels[index])

        return img, y, index



class InfluenceLogger:
    def __init__(self, queue: Queue = None, verbose: int = 1):
        """
        Args:
            queue (Queue): multiprocessing queue for inter-process logging.
            verbose (int): verbosity level (0=critical, 1=info, 2=debug)
        """
        self.queue = queue
        self.verbose = verbose
        self.start_time = time.time()

    def log(self, message: str, level: int = 1):
        """
        Log a message if it meets the verbosity threshold.

        Args:
            message (str): The message to log
            level (int): Message verbosity (0=critical, 1=info, 2=debug)
        """
        if level > self.verbose:
            return  # Skip messages above current verbosity

        pid = os.getpid()
        process_name = current_process().name
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time
        log_msg = (
            f"[{current_time}] (+{elapsed:.2f}s) "
            f"[PID {pid}] [{process_name}] {message}"
        )
        if self.queue:
            # Send to main process if running in a worker
            self.queue.put(log_msg)
        else:
            print(log_msg, flush=True)

    def reset_timer(self):
        """Reset the start time for elapsed time calculation."""
        self.start_time = time.time()


def log_listener(queue: Queue, verbose: int = 1):
    """Continuously listen for logs from workers and print them."""
    while True:
        try:
            msg = queue.get()
            if msg == "STOP":
                break
            print(msg, flush=True)
        except Exception as e:
            print(f"Logger error: {e}", file=sys.stderr)


        
class ImageFolderWithIndex(datasets.ImageFolder):
    """Custom dataset that includes image index as output. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        img, y = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        return img, y,index
    

    
def genloaders_fromfolder(train_dir, test_dir, loader_params):
    train_data =  ImageFolderWithIndex(train_dir, transform = loader_params.transform)
    test_data  =  ImageFolderWithIndex(test_dir, transform = loader_params.transform)
    
    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=loader_params['batch_size'],
        # pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    testloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=loader_params['batch_size'],
        # pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    IG_trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=loader_params['IG_batch_size'],
        # pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    return trainloader, testloader, IG_trainloader
    
    

def genloaders(X_train, y_train, X_test, y_test, loader_params):
    
    if loader_params['convert_to_torch']:
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train)
        
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test)
    
    if loader_params['training_size'] != 'full':
        X_train = X_train[0: loader_params['training_size']]
        y_train = y_train[0: loader_params['training_size']]
    
    if loader_params['conversion'] == 'rank':
        X_train, params = rank_convert_data(X_train)
        X_test = rank_convert_data(X_test, params)
        
    elif loader_params['conversion'] == 'uniform':
        # X = uniform_convert_data(X)
        X_train, params = uniform_convert_data(X_train)
        X_test = uniform_convert_data(X_test, params)
        
    elif loader_params['conversion'] == 'uniform_scale':
        # X = uniform_convert_data(X)
        X_train, params = uniform_scale_convert_data(X_train)
        X_test = uniform_scale_convert_data(X_test, params)
        
    elif loader_params['conversion'] == 'normalize':
        # X = normalized_convert_data(X)
        X_train, params = normalized_convert_data(X_train)
        X_test = normalized_convert_data(X_test, params)
        
    if loader_params['add_singleton']:
        X_train = X_train.unsqueeze(2).unsqueeze(3)
        X_test = X_test.unsqueeze(2).unsqueeze(3)
        
    my_dataset = Dataset_v2(X_train, y_train, loader_params['transform'])
    my_dataset_test = Dataset_v2(X_test, y_test, loader_params['transform'])

    trainloader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=loader_params['batch_size'],
        # pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    testloader = torch.utils.data.DataLoader(
        my_dataset_test,
        batch_size=loader_params['batch_size'],
        # pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    IG_trainloader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=loader_params['IG_batch_size'],
        # pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    return trainloader, testloader, IG_trainloader



def gen_pruned_loaders(X_train, y_train, X_test, y_test, loader_params):
    
    if loader_params['convert_to_torch']:
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train)
        
    if loader_params['training_size'] != 'full':
        X_train = X_train[0: loader_params['training_size']]
        y_train = y_train[0: loader_params['training_size']]
        
    if loader_params['train_indices'] != 'all':
        X_train = X_train[loader_params['train_indices']]
        y_train = y_train[loader_params['train_indices']]
    
    if loader_params['conversion'] == 'rank':
        X_train, params = rank_convert_data(X_train)
        X_test = rank_convert_data(X_test, params)
        
    elif loader_params['conversion'] == 'uniform':
        # X = uniform_convert_data(X)
        X_train, params = uniform_convert_data(X_train)
        X_test = uniform_convert_data(X_test, params)
        
    elif loader_params['conversion'] == 'uniform_scale':
        # X = uniform_convert_data(X)
        X_train, params = uniform_scale_convert_data(X_train)
        X_test = uniform_scale_convert_data(X_test, params)
        
    elif loader_params['conversion'] == 'normalize':
        # X = normalized_convert_data(X)
        X_train, params = normalized_convert_data(X_train)
        X_test = normalized_convert_data(X_test, params)
        
    if loader_params['add_singleton']:
        X_train = X_train.unsqueeze(2).unsqueeze(3)
        X_test = X_test.unsqueeze(2).unsqueeze(3)
        
    my_dataset = Dataset_v2(X_train, y_train,loader_params['transform'])
    my_dataset_test = Dataset_v2(X_test, y_test,loader_params['transform'])

    trainloader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=loader_params['batch_size'],
        # pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    testloader = torch.utils.data.DataLoader(
        my_dataset_test,
        batch_size=loader_params['batch_size'],
        # pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    return trainloader, testloader



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def test_model(model, testloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model = model.eval()
    correct = torch.tensor(0)

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels, indices = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            all_outs = model(inputs)
            predicted = torch.argmax(all_outs,1)
            correct = correct + torch.sum((predicted == labels).float())
    
    accuracy = float(correct) / float(len(testloader.dataset.labels))
    return accuracy



def get_labelwise_loaders(IG_trainloader, loader_params):
    label_to_indices = defaultdict(list)

    for idx in range(len(IG_trainloader.dataset)):
        _, label, _ = IG_trainloader.dataset[idx]
        label_to_indices[int(label)].append(idx)

    label_to_loader = {}
    for label, indices in label_to_indices.items():
        subset = Subset(IG_trainloader.dataset, indices)
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=loader_params['IG_batch_size'],
            num_workers=loader_params['num_workers']
        )
        label_to_loader[label] = loader
        
    return label_to_loader



def get_learning_config(model, train_params, config=None):
    options = []
    if config == 'pretrained_VGG16':
        # For vgg16, we start with a learning rate of 1e-3 for the last layer, and
        # decay it to 1e-7 at the first conv layer. The intermediate rates are
        # decayed linearly.
        lr = 0.0001
        options.append({
            'params': model.classifier.parameters(),
            'lr': lr,
        })
        final_lr = lr / 1000.0
        diff_lr = final_lr - lr
        lr_step = diff_lr / 44.0
        for i in range(43, -1, -1):
            options.append({
                'params': model.features[i].parameters(),
                'lr': lr + lr_step * (44-i)
            })

        optimizer = torch.optim.Adam(options, lr=1e-8)
        # Every 2 steps reduce the LR to 70% of the previous value.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
            
    elif config == 'pretrained_Resnet50':
        # For the resnet class of models, we decay the LR exponentially and reduce
        # it to a third of the previous value at each step.
        layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
        lr = 0.0001
        for layer_name in reversed(layers):
            options.append({
                "params": getattr(model, layer_name).parameters(),
                'lr': lr,
            })
            lr = lr / 3.0

        optimizer = optim.Adam(options, lr=1e-8)
        # Every 2 steps reduce the LR to 70% of the previous value.
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

    else:
        if train_params['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                model.parameters(),
                lr=train_params['init_rate'],
                momentum=0.9,
                weight_decay=train_params['weight_decay']
            )
        elif train_params['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=train_params['init_rate'],
                weight_decay=train_params['weight_decay']
            )
        elif train_params['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=train_params['init_rate'],
                weight_decay=train_params['weight_decay']
            )
            
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        if train_params['scheduler']['name'] == 'StepLR': 
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=train_params['scheduler']['step_size'],
                gamma= train_params['scheduler']['gamma']
            )
        elif train_params['scheduler']['name'] == 'MultiStepLR': 
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=train_params['scheduler']['milestones'],
                gamma= train_params['scheduler']['gamma']
            )
        elif train_params['scheduler']['name'] == 'CyclicLR':
            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer,
                train_params['init_rate'],
                train_params['scheduler']['max_lr'],
                train_params['scheduler']['step_size'],
                step_size_down=train_params['scheduler']['step_size'],
                mode='triangular',
                gamma=train_params['scheduler']['gamma']
            )

    if train_params['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
        
    elif train_params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        
    elif train_params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss()
        
    return optimizer, scheduler, criterion



def update_IG(IG, main_model, batch_indices, old_trainloss, IG_trainloader, train_params, influence_params, logger=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'

    if logger is not None: logger.log("Updating influence graph...", level=2)
    
    model = main_model
    model = model.to(device)
    model = model.eval()
    
    batch_indices = batch_indices.cpu()
    old_batchloss = old_trainloss[batch_indices]
    
    if train_params['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')
        
    elif train_params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        
    elif train_params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss(reduction = 'none')

    trainloss = np.zeros(IG.node_size)

    # Forward pass for full dataset
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

            trainloss[indices] = loss.detach().to(device)

            # Free up memory proactively
            del inputs, labels, outputs, loss
            if use_amp: torch.cuda.empty_cache()

    # Convert trainloss for numpy operations
    trainloss = trainloss.cpu().numpy()
    batchloss_diff = old_batchloss - trainloss[batch_indices]
    trainloss_diff = old_trainloss - trainloss
    
    if influence_params['clip_outliers']:
        mean_loss = np.mean(trainloss_diff)
        std_loss = np.std(trainloss_diff)
        lower_bound = mean_loss - 4 * std_loss
        upper_bound = mean_loss + 4 * std_loss
        
        trainloss_diff = np.clip(trainloss_diff, lower_bound, upper_bound)
        batchloss_diff = np.clip(batchloss_diff, lower_bound, upper_bound)
        
    if influence_params['loss_scaling_type'] is not None:
        
        if influence_params['set_zero_mean'] == 'full':
            batchloss_diff = batchloss_diff - np.mean(trainloss_diff)
            trainloss_diff = trainloss_diff - np.mean(trainloss_diff)
            
        if influence_params['set_zero_mean'] == 'separate':
            mask = torch.ones(trainloss_diff.size, dtype=torch.bool)
            mask[batch_indices] = False
            batchloss_diff = batchloss_diff - np.mean(batchloss_diff)
            trainloss_diff[mask.cpu()] = trainloss_diff[mask.cpu()] - np.mean(trainloss_diff[mask.cpu()])
            trainloss_diff[batch_indices] = batchloss_diff

        
        if influence_params['loss_scaling_span'] == 'batch':
            scale_ref = copy.copy(batchloss_diff)
        elif influence_params['loss_scaling_span'] == 'full':
            scale_ref = copy.copy(trainloss_diff)

        
        if influence_params['loss_scaling_type'] == 'mean_absolute':
            scale_ref = np.mean(np.abs(scale_ref)) 
            batchloss_diff = batchloss_diff / scale_ref
            trainloss_diff = trainloss_diff / scale_ref
            
        elif influence_params['loss_scaling_type'] == 'mean':
            scale_ref = np.abs(np.mean(scale_ref)) 
            batchloss_diff = batchloss_diff / scale_ref
            trainloss_diff = trainloss_diff / scale_ref
            
        elif influence_params['loss_scaling_type'] == 'max_absolute':
            scale_ref = np.max(np.abs(scale_ref)) 
            batchloss_diff = batchloss_diff / scale_ref
            trainloss_diff = trainloss_diff / scale_ref
            
        elif influence_params['loss_scaling_type'] == 'root_mean_squared':
            scale_ref = np.sqrt(np.mean(scale_ref**2)) 
            batchloss_diff = batchloss_diff / scale_ref
            trainloss_diff = trainloss_diff / scale_ref
            
        elif influence_params['loss_scaling_type'] == 'separated_rmse':
            mask = torch.ones(trainloss_diff.size, dtype=torch.bool)
            mask[batch_indices] = False
            scale_ref_batch = np.sqrt(np.mean(batchloss_diff**2))
            scale_ref_train = np.sqrt(np.mean(trainloss_diff[mask.cpu()]**2))
            batchloss_diff = batchloss_diff / scale_ref_batch
            trainloss_diff = trainloss_diff / scale_ref_train
            
        elif influence_params['loss_scaling_type'] == 'separated_absolute':
            mask = torch.ones(trainloss_diff.size, dtype=torch.bool)
            mask[batch_indices] = False
            scale_ref_batch = (np.mean(np.abs(batchloss_diff)))
            scale_ref_train = (np.mean(np.abs(trainloss_diff[mask.cpu()])))
            batchloss_diff = batchloss_diff / scale_ref_batch
            trainloss_diff = trainloss_diff / scale_ref_train
            trainloss_diff[batch_indices] = batchloss_diff
    
    IG.update_influence_graph(batch_indices, batchloss_diff, trainloss_diff)
    model = model.train()

    if logger is not None: logger.log("Updated influence graph. Returning to training...", level=2)
    return trainloss



def estimate_starting_trainloss(model, IG_trainloader, train_params, logger=None):
    
    # Detect device and mixed precision
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'

    if logger is not None: logger.log("Estimating starting trainloss...", level=1)

    model = model.to(device)
    model = model.eval()

    if train_params['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')
        
    elif train_params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        
    elif train_params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss(reduction = 'none')

    trainloss = torch.zeros(IG_trainloader.dataset.inputs.shape[0], device=device)

    # Enable AMP only if CUDA is available
    with torch.autocast(device_type=device):
        for inputs, labels, indices in IG_trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            allouts = model(inputs)
            loss = criterion(allouts, labels.long())
            
            # Detach before storing in trainloss
            trainloss[indices] = loss.detach()

    trainloss = trainloss.cpu().numpy()

    if logger is not None: logger.log("Estimated starting trainloss. Returning to compute influence graph...", level=1)
    
    model = model.train()
    return trainloss
    


def train_model_general(model, trainloader, train_params, config=None, logger=None):
    
    # Detect device and mixed precision
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'

    if logger is not None:
        logger.log("Training model generally with total params: {}...".format(count_parameters(model)), level=1)

    model = model.to(device)
    model = model.train()
    
    optimizer, scheduler, criterion = get_learning_config(model, train_params, config=config)

    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    for epoch in range(train_params['total_epochs']):
        
        if train_params['disp_epoch'] == True and logger is not None:
            logger.log("Starting epoch: {}...".format(epoch), level=1)
            logger.log("Accuracy: {}...".format(test_model(model, trainloader)), level=1)
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels, indices = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            if logger is not None:
                logger.log("Training model with trainloader {} of length {}...".format(i, len(inputs)), level=2)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                # Mixed precision on GPU
                with torch.autocast(device_type=device):
                    allouts = model(inputs)
                    loss = criterion(allouts, labels.long())
    
                # Scale gradients for FP16
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 on CPU
                allouts = model(inputs)
                loss = criterion(allouts, labels.long())
                loss.backward()
                optimizer.step()
                
        scheduler.step()
      
    if train_params['disp_accuracy_final'] == True and logger is not None:
        logger.log("Accuracy: {}...".format(test_model(model, trainloader)), level=1)
    
    model = model.eval()
    return model



def estimate_influencegraph(model, trainloader, IG_trainloader, train_params, influence_params, loader_params, config=None, logger=None):
    
    # Detect device and mixed precision
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'

    if logger is not None:
        logger.log("Estimating influence graph for model with total params: {}...".format(count_parameters(model)), level=1)
    
    model_IG = influence_params['graph_type'](
        trainloader.dataset.inputs.shape[0],
        trainloader.dataset.labels.squeeze().cpu().numpy(),
        loader_params['batch_size'],
        influence_params
    )
    
    model = model.to(device)
    model = model.train()
    
    trainloss = estimate_starting_trainloss(model, IG_trainloader, train_params, logger=logger)

    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    optimizer, scheduler, criterion = get_learning_config(model, train_params, config=config)

    for epoch in range(train_params['total_epochs']):
        
        if train_params['disp_epoch'] == True and logger is not None:
            logger.log("Starting epoch: {}...".format(epoch), level=1)
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels, indices = data
            inputs, labels = inputs.to(device), labels.to(device)

            if logger is not None:
                logger.log("Estimating influence graph with trainloader {} of length {}...".format(i, len(inputs)), level=2)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                # Mixed precision on GPU
                with torch.autocast(device_type=device):
                    allouts = model(inputs)
                    loss = criterion(allouts, labels.long())
    
                # Scale gradients for FP16
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 on CPU
                allouts = model(inputs)
                loss = criterion(allouts, labels.long())
                loss.backward()
                optimizer.step()
            
            trainloss = update_IG(model_IG, model, indices, trainloss, IG_trainloader, train_params, influence_params, logger=logger)
    
        scheduler.step()

        # Periodic GC to avoid hidden leaks
        if epoch % 5 == 0:
            gc.collect()
            if use_amp: torch.cuda.empty_cache()
      
    if train_params['disp_accuracy_final'] == True and logger is not None:
        logger.log("Accuracy: {}...".format(test_model(model, trainloader)), level=1)
    
    model = model.eval()
    return model, model_IG








