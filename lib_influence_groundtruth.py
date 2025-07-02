# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 20:33:44 2025

@author: User
"""

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import itertools 
import os
import pickle
# from scipy.sparse.csgraph import connected_components
import time 
# import scipy 
import torch.optim as optim

from lib_train import * 
import torch.nn as nn



# influence_GT_params: 

    

class InfluenceGraph_GT:

    def __init__(self, node_size,node_labels,batch_update_size, influence_params):
        self.node_size = node_size
        self.node_labels = node_labels
        self.node_list = list(range(0,node_size))
        
        self.locations = np.array(list(itertools.product(np.arange(0,batch_update_size),self.node_list)))
        self.locations = self.locations.reshape((batch_update_size,self.node_size,2))
        
        self.intraclass_only = influence_params['intraclass_only']
        
        row = np.array([0])
        col = np.array([0])
        data = np.array([0],dtype = influence_params['dtype'])
        self.influence_sum_mat = csr_matrix((data, (row, col)), shape = (node_size,node_size))
        self.influence_count_mat = csr_matrix((data, (row, col)), shape = (node_size,node_size))
        
    def update_normalized_graph(self):
        self.normgraph_mat = self.influence_sum_mat.multiply(self.influence_count_mat.power(-1.))
        return self.normgraph_mat
        
    def update_graph_mat_oneshot(self, locations_x,locations_y, influences):
        
        #  flattening in row-major order: (iterate through all columns for each row)
        #  Assume weights are already flattened
        
        influence_mat = csr_matrix((influences, (locations_x,locations_y)), shape = (self.node_size,self.node_size))
        count_mat = csr_matrix((np.ones(len(locations_x)), (locations_x,locations_y)), shape = (self.node_size,self.node_size))

        self.influence_sum_mat  = self.influence_sum_mat + influence_mat
        self.influence_count_mat = self.influence_count_mat + count_mat
        
    def update_influence_graph(self, batch_indices,trainlosses):
        
        batch_indices = batch_indices.cpu()
        
        mask_mat = np.ones((len(batch_indices),self.node_size))
        mask_mat[:,batch_indices] = 0 
        
        if self.intraclass_only:
            batchlabel_mat = np.tile(np.expand_dims(self.node_labels[batch_indices],1), (1, self.node_size))
            reflabel_mat = np.tile(self.node_labels, (len(batch_indices),1))
            labeleq_mat = batchlabel_mat == reflabel_mat
            mask_mat = mask_mat * labeleq_mat 
        
        influence_vec = trainlosses 
        influence_mat = np.tile(influence_vec, (len(batch_indices),1))
        
        # batchlossdiff_mat = np.tile(batch_lossdiff + epsilon,(1,self.node_size))  
        
        locations_temp = self.locations[mask_mat==1,:]
        influences = influence_mat[mask_mat==1]
        
        self.update_graph_mat_oneshot(
            batch_indices[locations_temp[:,0]],
            locations_temp[:,1],
            influences
        )
        
        
    def store_graph(self,folder,loader_params, influence_GT_params, influence_GT_train_params):
        self.update_normalized_graph()
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.chdir(folder)
        
        newpath = loader_params['dataset_name'] +'_GT' 
                           
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        os.chdir(newpath)
            
        files = os.listdir()
        
        ID = str(1+int(len(files)/2))
        sparse.save_npz(ID+".npz", self.normgraph_mat)
        with open(ID +'.pkl', 'wb') as handle:
            pickle.dump([loader_params, influence_GT_params, influence_GT_train_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        os.chdir('..')
        os.chdir('..')
        
        return
    
    def load_graph(self,folder, dataset_name,ID):
        self.update_normalized_graph()
        os.chdir(folder)
        
        os.chdir(dataset_name+'_GT')
        if ID == 'latest':
            files = os.listdir()
            ID = str(int(len(files)/2))
            print("File ID:", ID)
            
        self.normgraph_mat = sparse.load_npz(str(ID)+".npz")
        
        os.chdir('..')
        os.chdir('..')
        
        return self.normgraph_mat
    
    
    
def update_InfluenceGraph_GT(model, IG_GT, batch_indices, IG_trainloader, train_params):
    # s = time.time() 
    
    model = model.eval()
    
    dataiter = iter(IG_trainloader)
    trainloss = np.zeros(IG_GT.node_size)
    
    if train_params['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')
    elif train_params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(reduction = 'none') 
    elif train_params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss(reduction = 'none') 
        
        # one batch update
    # s = time.time()
    with torch.no_grad():
        for i, data in enumerate(IG_trainloader, 0):
            # get the inputs
            inputs, labels, indices = data
            allouts = model(inputs)
            loss = criterion(allouts, labels.long())
            trainloss[indices.cpu()] = loss.cpu() 
            
    # print(time.time()-s)
    
    IG_GT.update_influence_graph(batch_indices,trainloss)
    model = model.train()
    return IG_GT, np.mean(trainloss[batch_indices.cpu()])


def batch_influence_GT(model_params, trainloader, IG_trainloader, influence_GT_params, influence_GT_train_params, loader_params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    node_size = trainloader.dataset.inputs.shape[0]
    
    IG_GT = InfluenceGraph_GT(
        node_size,
        trainloader.dataset.labels.squeeze().cpu().numpy(),
        trainloader.batch_size,influence_GT_params
    )
        
    if influence_GT_train_params['criterion'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif influence_GT_train_params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss() 
    elif influence_GT_train_params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss() 
        
    flag = 0
    
    for epoch in range(influence_GT_params['training_iterations']):
        print('Training iteration:', epoch)
        for i, data in enumerate(trainloader, 0):
            inputs, labels, indices = data
                
            model = model_params['type'](model_params['name'], model_params['in_channels'], batchnorm = model_params['batchnorm'])
            
            # trainloss = estimate_starting_trainloss(model, IG_trainloader, train_params)
            if influence_GT_train_params['optimizer'] == 'SGD':
                    optimizer = optim.SGD(
                        model.parameters(),
                        lr=influence_GT_train_params['init_rate'],
                        momentum=0.9,
                        weight_decay=influence_GT_train_params['weight_decay']
                    )
                
            elif influence_GT_train_params['optimizer'] == 'Adam':
                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=influence_GT_train_params['init_rate'],
                        weight_decay=influence_GT_train_params['weight_decay']
                    )
                
            elif influence_GT_train_params['optimizer'] == 'AdamW':
                    optimizer = optim.AdamW(
                        model.parameters(),
                        lr=influence_GT_train_params['init_rate'],
                        weight_decay=influence_GT_train_params['weight_decay']
                    )
            
            scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
            
            if influence_GT_train_params['scheduler']['name'] == 'StepLR': 
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=influence_GT_train_params['scheduler']['step_size'], gamma= influence_GT_train_params['scheduler']['gamma'])
            elif influence_GT_train_params['scheduler']['name'] == 'MultiStepLR': 
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=influence_GT_train_params['scheduler']['milestones'], gamma= influence_GT_train_params['scheduler']['gamma'])
            elif influence_GT_train_params['scheduler']['name'] == 'CyclicLR':
                scheduler = optim.lr_scheduler.CyclicLR(optimizer, influence_GT_train_params['init_rate'], influence_GT_train_params['scheduler']['max_lr'], 
                        influence_GT_train_params['scheduler']['step_size'], 
                        step_size_down=influence_GT_train_params['scheduler']['step_size'], 
                        mode='triangular', gamma=influence_GT_train_params['scheduler']['gamma'])
            
            model = model.to(device)
            model = model.train()
            
            s = time.time()
            
            for mini_epoch in range(influence_GT_train_params['total_epochs']):
                if influence_GT_train_params['disp_epoch'] == True: 
                    print('epoch: ' + str(epoch))
                
                optimizer.zero_grad()
                allouts = model(inputs)
                
                loss = criterion(allouts, labels.long()) 
                # print(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                
            if influence_GT_train_params['disp_time_per_batch'] == True and flag == 0: 
                print("Time for one batch update:",time.time()-s)
                flag = 1
                
            IG_GT, mean_trainloss = update_InfluenceGraph_GT(model, IG_GT, indices, IG_trainloader, influence_GT_train_params)
            
            if influence_GT_train_params['disp_loss_epoch'] == True:
                print("Training Loss:", mean_trainloss)
            
    IG_GT.update_normalized_graph()
    
    return IG_GT
            
                


# def representative_influence_GT(model, trainloader, influence_GT_params, influence_GT_train_params):
    
    




