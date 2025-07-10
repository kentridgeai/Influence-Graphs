# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:48:11 2025

@author: User
"""

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


from torch.utils import data

import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
import torchvision
from lib_graph import * 
from lib_preprocessing import *
import matplotlib.pyplot as plt
import copy



class Dataset_v2(data.Dataset):
    # Characterizes a dataset for PyTorch
    
    def __init__(self, inputs, labels, transform=None):
        # 'Initialization'
        self.labels = labels
        # self.list_IDs = list_IDs
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
        shuffle=True,
        pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    testloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=loader_params['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    IG_trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=loader_params['IG_batch_size'],
        shuffle=True,
        pin_memory=True,
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
        shuffle=True,
        pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    testloader = torch.utils.data.DataLoader(
        my_dataset_test,
        batch_size=loader_params['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    IG_trainloader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=loader_params['IG_batch_size'],
        shuffle=True,
        pin_memory=True,
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
        shuffle=True,
        pin_memory=True,
        num_workers=loader_params['num_workers']
    )
    testloader = torch.utils.data.DataLoader(
        my_dataset_test,
        batch_size=loader_params['batch_size'],
        shuffle=False,
        pin_memory=True,
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
    dataiter = iter(testloader)

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



def update_IG(IG, main_model, batch_indices, old_trainloss, IG_trainloader, train_params, influence_params):
    # s = time.time() 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    old_trainloss = copy.deepcopy(old_trainloss)
    
    model = copy.deepcopy(main_model)
    model = model.to(device)
    model = model.eval()
    
    dataiter = iter(IG_trainloader)
    trainloss = np.zeros(IG.node_size)
    batch_indices = batch_indices.cpu()
    old_batchloss = old_trainloss[batch_indices]
    
    if train_params['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')
        
    elif train_params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        
    elif train_params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss(reduction = 'none') 
        
    with torch.no_grad():
        for i, data in enumerate(IG_trainloader, 0):
            # get the inputs
            inputs, labels, indices = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            allouts = model(inputs)
            loss = criterion(allouts, labels.long())
            trainloss[indices.cpu()] = loss.cpu() 
            
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
        
        # print(np.max(trainloss_diff),np.min(trainloss_diff))
        
    # print("diff:", np.mean(batchloss_diff)-np.mean(trainloss_diff))
    # loss_sum = np.mean(batchloss_diff)-np.mean(trainloss_diff)
    
    IG.update_influence_graph(batch_indices, batchloss_diff , trainloss_diff)
    model = model.train()
    
    # return trainloss, batchloss_diff, trainloss_diff
    # return trainloss, loss_sum
    return trainloss



def estimate_starting_trainloss(model, IG_trainloader, train_params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model = model.eval()
    trainloss = np.zeros(IG_trainloader.dataset.inputs.shape[0])
    
    if train_params['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')
        
    elif train_params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        
    elif train_params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss(reduction = 'none')
        
    with torch.no_grad():
        for i, data in enumerate(IG_trainloader, 0):
            # get the inputs
            inputs, labels, indices = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            allouts = model(inputs)
            loss = criterion(allouts, labels.long())
            trainloss[indices.cpu()] = loss.cpu() 
            
    model = model.train()
    return trainloss
    


def train_model_general(model, trainloader, train_params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Total Model Params: ", count_parameters(model))

    model = model.to(device)
    model = model.train()
    
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

    init_epoch = 0
    all_train_losses = []
    train_loss_min = 9999
    
    if train_params['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif train_params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss() 
    elif train_params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss() 
    
    flag = 0
    
    for epoch in range(train_params['total_epochs']):

        # batchloss_diffs = np.empty(0)
        # trainloss_diffs = np.empty(0)
        # loss_sums = 0
        # loss_change = []
        s = time.time()
        
        if train_params['disp_epoch'] == True: 
            print('epoch: ' + str(epoch))
        
        train_loss = []
        loss_weights = [] 
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels, indices = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            allouts = model(inputs)
            
            loss = criterion(allouts, labels.long()) 
            # print(loss.item())
            loss.backward()
            train_loss.append(loss.item())
            
            loss_weights.append(len(labels))
            optimizer.step()
            
        scheduler.step()
        all_train_losses.append(np.average(np.array(train_loss), weights=np.array(loss_weights)))
        if train_params['disp_loss_epoch'] == True:
            print("Training Loss:", all_train_losses[-1])
        
        if train_params['disp_time_per_epoch'] == True and flag == 0: 
            print("Time for one epoch:", time.time()-s)
            flag = 1   
        
    if train_params['disp_loss_final'] == True:
        print(all_train_losses[-1])
      
    if train_params['disp_accuracy_final'] == True:
        accuracy = test_model(model, trainloader)
    
    model = model.eval()
    
    return model, all_train_losses



def estimate_influencegraph(model, trainloader, IG_trainloader, train_params, influence_params, loader_params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Total Model Params: ", count_parameters(model))
    
    model_IG = influence_params['graph_type'](
        trainloader.dataset.inputs.shape[0],
        trainloader.dataset.labels.squeeze().cpu().numpy(),
        loader_params['batch_size'],
        influence_params
    )
    
    model = model.to(device)
    model = model.train()
    
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

    init_epoch = 0
    all_train_losses = []
    train_loss_min = 9999
    
    if train_params['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
        
    elif train_params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        
    elif train_params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss()
    
    flag = 0
    
    trainloss = estimate_starting_trainloss(model, IG_trainloader, train_params)
    # plt.hist(trainloss)
    # plt.pause(1)
    
    for epoch in range(train_params['total_epochs']):

        # batchloss_diffs = np.empty(0)
        # trainloss_diffs = np.empty(0)
        # loss_sums = 0
        # loss_change = []
        s = time.time()
        
        if train_params['disp_epoch'] == True: 
            print('epoch: ' + str(epoch))
         
        train_loss = []
        loss_weights = [] 
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels, indices = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            allouts = model(inputs)
            
            loss = criterion(allouts, labels.long()) 
            # print(loss.item())
            loss.backward()
            train_loss.append(loss.item())
            
            loss_weights.append(len(labels))
            optimizer.step()
            
            trainloss = update_IG(model_IG, model, indices, trainloss, IG_trainloader, train_params, influence_params)
            # loss_sums = loss_sums + loss_sum
            # batchloss_diffs = np.append(batchloss_diffs,batch_diffs)
            # trainloss_diffs = np.append(trainloss_diffs,trainloss_diff)
            
            # model = model.train()
        
        # plt.hist(trainloss)
        # plt.pause(1)
        
        # plt.hist(batchloss_diffs,20)
        # print(np.mean(batchloss_diffs))
        # print(np.mean(trainloss_diffs))
        # print(loss_sums)
        
        # plt.pause(1)
    
        scheduler.step()
        all_train_losses.append(np.average(np.array(train_loss),weights=np.array(loss_weights)))
        if train_params['disp_loss_epoch'] == True:
            print("Training Loss:", all_train_losses[-1])
        
        if train_params['disp_time_per_epoch'] == True and flag == 0: 
            print("Time for one epoch:",time.time()-s)
            flag = 1
            
    if train_params['disp_loss_final'] == True:
        print(all_train_losses[-1])
      
    if train_params['disp_accuracy_final'] == True:
        accuracy = test_model(model,trainloader)
    
    model = model.eval()
    # model_IG.update_normalized_graph()
    
    return model, all_train_losses, model_IG








