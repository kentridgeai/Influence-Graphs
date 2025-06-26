# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:52:26 2025

@author: User
"""

import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np 
from networkx.classes.function import *
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from copy import copy 
import random
from collections import defaultdict
from scipy.sparse import csr_matrix


# Convert to NetworkX graph


def IG_coverageprune_data(G, target_num_nodes=500, batch_remove=1,mode='max'):
    
    G_copy = G.copy()
    flag = 1
    nodes_list = np.arange(G_copy.shape[0])
    if mode == 'max':
        
        while flag == 1:        
            if len(nodes_list)<batch_remove+ target_num_nodes:
                batch_remove = len(nodes_list) - target_num_nodes
                flag = 0 
            
            M = G_copy.max(axis=0).A.flatten()
            top_k_indices = np.argpartition(-M, batch_remove)[:batch_remove]
            nodes_to_remove = top_k_indices
            
            mask = ~np.isin(np.arange(G_copy.shape[0]), nodes_to_remove)
            
            G_copy = G_copy[mask][:, mask]
            
            nodes_list = nodes_list[mask]
            
    elif mode == 'mean':
        
        while flag == 1:        
            if len(nodes_list)<batch_remove+ target_num_nodes:
                batch_remove = len(nodes_list) - target_num_nodes
                flag = 0 
                
            M = G_copy.mean(axis=0).A.flatten()
            top_k_indices = np.argpartition(-M, batch_remove)[:batch_remove]
            nodes_to_remove = top_k_indices
        
            mask = ~np.isin(np.arange(G_copy.shape[0]), nodes_to_remove)
            
            G_copy = G_copy[mask][:, mask]
            
            nodes_list = nodes_list[mask]
        
    return nodes_list, G_copy
        



def IG_iterativecoverageprune_data(Gerard, target_num_nodes=500, max_iterations=100):
    """
    Modified function to swap nodes based on in-degree criteria.
    
    Parameters:
        G (csr_matrix): Input graph in sparse matrix format.
        target_num_nodes (int): Desired number of nodes to retain.
        max_iterations (int): Number of iterations for swapping nodes.
        
    Returns:
        nodes_list (np.ndarray): Indices of the remaining nodes.
        G_copy (csr_matrix): Modified graph with remaining nodes.
    """
    # Create a copy of the graph and initialize variables
    G_copy = Gerard.copy()
    nodes_list = np.arange(G_copy.shape[0])
    nodes_list2 = np.arange(G_copy.shape[0])
    
    # Start with all nodes
    flag = 1
    batch_remove = 1
    while flag == 1:        
        if len(nodes_list)<batch_remove+ target_num_nodes:
            batch_remove = len(nodes_list) - target_num_nodes
            flag = 0 
            
        M = G_copy.mean(axis=0).A.flatten()
        top_k_indices = np.argpartition(-M, batch_remove)[:batch_remove]
        nodes_to_remove = top_k_indices
    
        mask = ~np.isin(np.arange(G_copy.shape[0]), nodes_to_remove)
        
        G_copy = G_copy[mask][:, mask]
        nodes_list = nodes_list[mask]
         
    indices = nodes_list 
    
    G = Gerard.copy()
    # Step 2: Iteratively swap nodes
    for _ in range(max_iterations):
    # Find element outside `indices` with the smallest in-degree (mean of G[inside_indices, j])
        outside_indices = np.setdiff1d(np.arange(G.shape[0]), indices)
        in_degrees_outside = G[np.ix_(indices, outside_indices)].mean(axis=0).A.flatten()
        min_in_degree_node = outside_indices[np.argmin(in_degrees_outside)]
    
        # Find element inside `indices` with the largest in-degree (mean of G[inside_indices, j], excluding self-connection)
        G_inside = G[np.ix_(indices, indices)].A  # Convert to dense array for easier masking
        np.fill_diagonal(G_inside, 0)  # Zero out self-connections
        N = float(len(indices))
        in_degrees_inside = G_inside.mean(axis=0)*(N/N-1)  # Compute mean ignoring self-connections
        max_in_degree_node = indices[np.argmax(in_degrees_inside)]
    
        # Swap
        indices = np.setdiff1d(indices, [max_in_degree_node])  # Remove the max node
        indices = np.append(indices, min_in_degree_node)
    
        # Swap
        indices = np.setdiff1d(indices, [max_in_degree_node])  # Remove the max node
        indices = np.append(indices, min_in_degree_node)  # Add the min node

    # Return the resulting indices and updated graph
    mask = np.isin(np.arange(Gerard.shape[0]), indices)
    G = G[mask][:, mask]
    nodes_list = nodes_list2[mask]  # Keep only nodes in top_indices

    return nodes_list, G



def IG_adversaryprune_data(G, target_num_nodes=500, batch_remove=1,mode='max'):
    
    G_copy = G.copy()
    flag = 1
    nodes_list = np.arange(G_copy.shape[0])
    if mode == 'max':
        
        while flag == 1:        
            if len(nodes_list)<batch_remove+ target_num_nodes:
                batch_remove = len(nodes_list) - target_num_nodes
                flag = 0 
            
            M = G_copy.max(axis=1).A.flatten()
            top_k_indices = np.argpartition(M, batch_remove)[:batch_remove]
            nodes_to_remove = top_k_indices
            
            mask = ~np.isin(np.arange(G_copy.shape[0]), nodes_to_remove)
            
            G_copy = G_copy[mask][:, mask]
            
            nodes_list = nodes_list[mask]
            
    elif mode == 'mean':
        
        while flag == 1:        
            if len(nodes_list)<batch_remove+ target_num_nodes:
                batch_remove = len(nodes_list) - target_num_nodes
                flag = 0 
                
            M = G_copy.mean(axis=1).A.flatten()
            top_k_indices = np.argpartition(M, batch_remove)[:batch_remove]
            nodes_to_remove = top_k_indices
        
            mask = ~np.isin(np.arange(G_copy.shape[0]), nodes_to_remove)
            
            G_copy = G_copy[mask][:, mask]
            
            nodes_list = nodes_list[mask]
               
    return nodes_list, G_copy

    