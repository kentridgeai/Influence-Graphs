# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:14:59 2025

@author: User
"""
import numpy  as np
import torch


def rank_convert_data(data):
    
    for i in range(data.shape[1]):
        temp,data[:,i,0,0] = torch.unique(data[:,i,0,0],return_inverse=True)
        data[:,i,0,0]  = data[:,i,0,0]/torch.max(data[:,i,0,0]) 
        # dataset.data = dataset.data
    
    return data


def uniform_convert_data(data,params=None):
    if params == None:
        params = [] 
        for i in range(data.shape[1]):
            params.append([torch.min(data[:,i,0,0]),torch.max(data[:,i,0,0])])
            if torch.max(data[:,i,0,0])>torch.min(data[:,i,0,0]):
                data[:,i,0,0] = (data[:,i,0,0] - torch.min(data[:,i,0,0])) / (torch.max(data[:,i,0,0]) - torch.min(data[:,i,0,0])) 
            # else:
            #     print('mambaaaaaa')
            # dataset.data = dataset.data
        return data,params
        
    else:
        print('here')
        for i in range(data.shape[1]):
            if params[i][1]>params[i][0]:
                data[:,i,0,0] = (data[:,i,0,0] - params[i][0]) / (params[i][1] - params[i][0]) 
        return data


def uniform_scale_convert_data(data,params=None):
    if params == None:
        params = [] 
        for i in range(data.shape[1]):
            params.append([torch.min(data[:,i,0,0]),torch.max(data[:,i,0,0])])
            if torch.max(data[:,i,0,0])>torch.min(data[:,i,0,0]):
                data[:,i,0,0] = (data[:,i,0,0]) / (torch.max(data[:,i,0,0]) - torch.min(data[:,i,0,0])) 
            # else:
            #     print('mambaaaaaa')
            # dataset.data = dataset.data
        return data,params
        
    else:
        print('here')
        for i in range(data.shape[1]):
            if params[i][1]>params[i][0]:
                data[:,i,0,0] = (data[:,i,0,0]) / (params[i][1] - params[i][0]) 
        return data
    

def normalized_convert_data(data,params=None):
    if params == None: 
        params = [] 
        for i in range(data.shape[1]):
            params.append([torch.mean(data[:,i,0,0]),torch.std(data[:,i,0,0])])
            if torch.std(data[:,i,0,0])>0:
                data[:,i,0,0] = (data[:,i,0,0] - torch.mean(data[:,i,0,0])) / (torch.std(data[:,i,0,0]))
            
        return data, params
    else:
        for i in range(data.shape[1]):
            if params[i][1]>0:
                data[:,i,0,0] = (data[:,i,0,0] - params[i][0]) /  (params[i][1])
        return data