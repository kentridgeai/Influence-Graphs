# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:12:02 2025

@author: User
"""

import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
from networkx.classes.function import *
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec



def vis_influencepairs(G, images, min_percentile=0, max_percentile=100, num_pairs=10):
    
    W = G.data
    x,y = G.nonzero()
    print(np.min(W), np.max(W))
    sorted_indices = np.argsort(W)
    
    # Filter edges based on weight percentiles
    min_weight_id = int(min_percentile*len(W)/100.0)
    max_weight_id = int(max_percentile*len(W)/100.0) 

    indices = sorted_indices[min_weight_id:max_weight_id]
    
    if max_percentile == 100:
        indices = indices[::-1]
        
        
    W = W[indices]
    x = x[indices]
    y = y[indices]
    
    
    grid_size = int(np.sqrt(num_pairs))
    fig = plt.figure(figsize=(6, 6))  # Keep the overall figure size the same
    outer_grid = gridspec.GridSpec(grid_size, grid_size, wspace=0.2, hspace=0.2)  # Adjust spacing
    
    for i in range(num_pairs):
        row = i // grid_size
        col = i % grid_size
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[row, col], wspace=0.01)  # Reduce inner spacing
        
        ax1 = fig.add_subplot(inner_grid[0])
        ax2 = fig.add_subplot(inner_grid[1])
        
        ax1.imshow(images[x[i]].permute(1, 2, 0).squeeze().cpu())
        ax2.imshow(images[y[i]].permute(1, 2, 0).squeeze().cpu())
        
        # Remove axis for a cleaner look
        ax1.axis('off')
        ax2.axis('off')
    
    plt.show()
    
    
    # for i in range(num_pairs):
    #     f, axarr = plt.subplots(1,2)
    #     axarr[0].imshow(images[L[i][0]].permute(1,2,0).squeeze().cpu())
    #     axarr[1].imshow(images[L[i][1]].permute(1,2,0).squeeze().cpu())
    #     # Create a bar with red color that spans the entire plot
        
    




def vis_influencenodes(G,images, min_percentile = 0, max_percentile = 100, num_nodes = 10):
    
    W = G.sum(axis=1)
    sorted_indices = np.argsort(W.flatten())[0]
    
    # Filter edges based on weight percentiles
    min_weight_id = int(min_percentile*np.prod(W.shape)/100.0)
    max_weight_id = int(max_percentile*np.prod(W.shape)/100.0) 

    indices = sorted_indices[0,min_weight_id:max_weight_id]
    
    if max_percentile == 100:
        indices = indices[::-1]
        
    
    grid_size = int(np.sqrt(num_nodes))
    f, axarr = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    
    for i in range(num_nodes):
        row = i // grid_size
        col = i % grid_size
        axarr[row, col].imshow(images[indices[0,i]].permute(1, 2, 0).squeeze().cpu())
        
        # Remove axis for a cleaner look
        axarr[row, col].axis('off')
    
    # Create a bar with red color that spans the entire plot, shifted upwards
    bar = patches.Rectangle((0, -0.02), 1, 0.05, transform=f.transFigure, color='red')
    f.patches.append(bar)
    
    # Green out the part between min_percentile and max_percentile
    green_bar = patches.Rectangle((min_percentile / 100, -0.02), (max_percentile - min_percentile) / 100, 0.05, transform=f.transFigure, color='green')
    f.patches.append(green_bar)
    
    plt.show()
    
    # for i in range(num_nodes):
    #     f, ax = plt.subplots()
    #     ax.imshow(images[A[i]].permute(1,2,0).squeeze().cpu())
        
    #     bar = patches.Rectangle((0, -0.03), 1, 0.05, transform=f.transFigure, color='red')
    #     f.patches.append(bar)
        
    #     # Green out the part between min_percentile and max_percentile
    #     green_bar = patches.Rectangle((min_percentile / 100, -0.03), (max_percentile - min_percentile) / 100, 0.05, transform=f.transFigure, color='green')
    #     f.patches.append(green_bar)
        
    #     # plt.title(f'min_percentile: {min_percentile}, max_percentile: {max_percentile}')
    #     plt.pause(1)
        


   
