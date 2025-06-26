# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:08:27 2025

@author: User
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import itertools 
import os
import pickle
from scipy.sparse.csgraph import connected_components
import time 
import networkx as nx
import scipy 


class InfluenceGraph:

    def __init__(self, node_size,node_labels,batch_update_size, influence_params):
        self.node_size = node_size
        self.node_labels = node_labels
        self.node_list = list(range(0,node_size))
        
        self.locations = np.array(list(itertools.product(np.arange(0,batch_update_size),self.node_list)))
        self.locations = self.locations.reshape((batch_update_size,self.node_size,2))
        
        self.class_normalize = influence_params['class_normalize']
        self.remove_negatives = influence_params['remove_negatives']
        self.clipping = influence_params['clipping']
        self.intraclass_only = influence_params['intraclass_only']
        self.negative_clipping = influence_params['negative_clipping']
        
        row = np.array([0])
        col = np.array([0])
        data = np.array([0],dtype = influence_params['dtype'])
        self.graph_mat = csr_matrix((data, (row, col)), shape = (node_size,node_size))
        self.graph_counts = csr_matrix((data, (row, col)), shape = (node_size,node_size))
        
    def update_normalized_graph(self):
        self.normgraph_mat = self.graph_mat.multiply(self.graph_counts.power(-1.))
        return self.normgraph_mat
    
    def update_graph_mat(self, row, cols, weights, class_normalizer = 1):
        self.graph_mat[row,cols] = self.graph_mat[row,cols] + weights
        self.graph_counts[row,cols] = self.graph_counts[row,cols] + class_normalizer*np.ones(len(cols))
        
    def update_graph_mat_oneshot(self, locations_x,locations_y, weights, class_normalizer = 1):
        
        #  flattening in row-major order: (iterate through all columns for each row)
        #  Assume weights are already flattened
        
        weight_mat = csr_matrix((weights, (locations_x,locations_y)), shape = (self.node_size,self.node_size))
        count_mat = csr_matrix((class_normalizer*np.ones(len(locations_x)), (locations_x,locations_y)), shape = (self.node_size,self.node_size))

        self.graph_mat  = self.graph_mat + weight_mat
        self.graph_counts = self.graph_counts + count_mat
        
        
    def update_influence_graph(self, batch_indices, batch_lossdiff, train_lossdiff):
        
        # Default params: class_normalize = False, remove_negatives = False, clipping = True, intraclass_only = True
        #  itertools also performs computations in row major order 
        batch_indices = batch_indices.cpu()
        epsilon = 0.00000001
        
        # locations = np.array(list(itertools.product(batch_indices,self.node_list)))
        # locations = locations.reshape((len(batch_indices),self.node_size,2))
        mask_mat = np.ones((len(batch_indices),self.node_size))
        
        mask_mat[:,batch_indices] = 0 
        
        if self.class_normalize:
            class_normalizer = len(batch_indices)
        else:
            class_normalizer = 1
        
        if self.intraclass_only:
            batchlabel_mat = np.tile(np.expand_dims(self.node_labels[batch_indices],1), (1, self.node_size))
            reflabel_mat = np.tile(self.node_labels, (len(batch_indices),1))
            labeleq_mat = batchlabel_mat == reflabel_mat
            mask_mat = mask_mat * labeleq_mat 
        
        weights_vec = train_lossdiff # old - new
        
        if self.remove_negatives:
            weights_vec = weights_vec * (weights_vec>0)
        
        weights_mat = np.tile(weights_vec, (len(batch_indices),1))
        
        # batchlossdiff_mat = np.tile(batch_lossdiff + epsilon,(1,self.node_size))
        batchlossdiff_mat = np.tile(np.expand_dims(batch_lossdiff,1) + epsilon,(1,self.node_size))
        
        weights_mat = weights_mat/batchlossdiff_mat
        
        if self.clipping:
            weights_mat[weights_mat>1] = 1
        
        if self.negative_clipping:
            weights_mat[weights_mat<-1] = -1
        
        locations_temp = self.locations[mask_mat==1,:]
        weights_flat = weights_mat[mask_mat==1]
        
        self.update_graph_mat_oneshot(
            batch_indices[locations_temp[:,0]],
            locations_temp[:,1],
            weights_flat,
            class_normalizer
        )
        
        
    def store_graph(self,folder,loader_params,influence_params,train_params):
        self.update_normalized_graph()
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.chdir(folder)
        
        newpath = loader_params['dataset_name'] 
                           
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        os.chdir(newpath)
            
        files = os.listdir()
        
        ID = str(1+int(len(files)/2))
        sparse.save_npz(ID+".npz", self.normgraph_mat)
        with open(ID +'.pkl', 'wb') as handle:
            pickle.dump([loader_params, influence_params, train_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        os.chdir('..')
        os.chdir('..')
        return
    
    def load_graph(self,folder, dataset_name,ID):
        self.update_normalized_graph()
        os.chdir(folder)
        
        os.chdir(dataset_name)
        if ID == 'latest':
            files = os.listdir()
            ID = str(int(len(files)/2))
            print("File ID:", ID)
            
        self.normgraph_mat = sparse.load_npz(str(ID)+".npz")
        
        os.chdir('..')
        os.chdir('..')
        
        return self.normgraph_mat



class InfluenceGraphv2: # Difference in difference based approaches 

    def __init__(self, node_size,node_labels,batch_update_size, influence_params):
        self.node_size = node_size
        self.node_labels = node_labels
        self.node_list = list(range(0,node_size))
        
        self.locations = np.array(list(itertools.product(np.arange(0,batch_update_size),self.node_list)))
        self.locations = self.locations.reshape((batch_update_size,self.node_size,2))
        
        self.class_normalize = influence_params['class_normalize']
        self.remove_negatives = influence_params['remove_negatives']
        self.clipping = influence_params['clipping']
        self.intraclass_only = influence_params['intraclass_only']
        self.negative_clipping = influence_params['negative_clipping']
        
        # self.C0 = 1+(float(batch_update_size - 1)/float(node_size - (2*batch_update_size)))
        # self.C1 = float((batch_update_size - 1)*(node_size - batch_update_size))/float(batch_update_size*(node_size- (2*batch_update_size)))
        
        self.C0 = 1
        self.C1 = 1
        row = np.array([0])
        col = np.array([0])
        data = np.array([0],dtype = influence_params['dtype'])
        self.influence_sum = csr_matrix((data, (row, col)), shape = (node_size,node_size))
        self.influence_counts = csr_matrix((data, (row, col)), shape = (node_size,node_size))
        self.passiveloss_sum  = np.zeros(self.node_size)
        self.passiveloss_counts  = np.zeros(self.node_size)
        self.activeloss_sum  = np.zeros(self.node_size)
        self.activeloss_counts  = np.zeros(self.node_size)
        
        
    def update_normalized_graph(self):
        
        passiveloss_mean = self.passiveloss_sum/self.passiveloss_counts
        activeloss_mean = self.activeloss_sum/self.activeloss_counts
        # epsilon = 0.1
        self.normgraph_mat = self.influence_sum.multiply(self.influence_counts.power(-1.))
        
        rows,cols = self.normgraph_mat.nonzero()
        
        passive_mat = csr_matrix((passiveloss_mean[cols], (rows,cols)), shape = (self.node_size,self.node_size))
        active_mat = csr_matrix((activeloss_mean[rows], (rows,cols)), shape = (self.node_size,self.node_size))
        
        self.normgraph_mat = self.normgraph_mat - passive_mat
        # self.normgraph_mat = self.normgraph_mat.multiply(active_mat.power(-1.))
        
        return self.normgraph_mat

        
    def update_graph_mat_oneshot(self, locations_x,locations_y, lossdiff, class_normalizer = 1):
        
        #  flattening in row-major order: (iterate through all columns for each row)
        #  Assume weights are already flattened
        
        lossdiff_mat = csr_matrix((lossdiff, (locations_x,locations_y)), shape = (self.node_size,self.node_size))
        count_mat = csr_matrix((class_normalizer*np.ones(len(locations_x)), (locations_x,locations_y)), shape = (self.node_size,self.node_size))

        self.influence_sum  = self.influence_sum + lossdiff_mat
        self.influence_counts = self.influence_counts + count_mat
        
        
    def update_influence_graph(self, batch_indices, batch_lossdiff, train_lossdiff):
        
        # Default params: class_normalize = False, remove_negatives = False, clipping = True, intraclass_only = True
        #  itertools also performs computations in row major order 
        batch_indices = batch_indices.cpu()
        epsilon = 0.00000001
        
        # locations = np.array(list(itertools.product(batch_indices,self.node_list)))
        # locations = locations.reshape((len(batch_indices),self.node_size,2))
        mask_mat = np.ones((len(batch_indices),self.node_size))
        
        mask_mat[:,batch_indices] = 0 
        
        if self.class_normalize:
            class_normalizer = len(batch_indices)
        else:
            class_normalizer = 1
            
        if self.intraclass_only:
            batchlabel_mat = np.tile(np.expand_dims(self.node_labels[batch_indices],1), (1, self.node_size))
            reflabel_mat = np.tile(self.node_labels, (len(batch_indices),1))
            labeleq_mat = batchlabel_mat == reflabel_mat
            mask_mat = mask_mat * labeleq_mat 
        
            # old - new
        
        if self.remove_negatives:
            train_lossdiff = train_lossdiff * (train_lossdiff>0)
        
        trainlossdiff_mat = np.tile(train_lossdiff, (len(batch_indices),1))
        
        # batchlossdiff_mat = np.tile(batch_lossdiff + epsilon,(1,self.node_size))
        # batchlossdiff_mat = np.tile(np.expand_dims(batch_lossdiff,1) + epsilon,(1,self.node_size))
        
        
        if self.clipping:
            trainlossdiff_mat[trainlossdiff_mat>1] = 1
        
        if self.negative_clipping:
            trainlossdiff_mat[trainlossdiff_mat<-1] = -1
        
        locations_temp = self.locations[mask_mat==1,:]
        trainlossdiff_flat = self.C0*trainlossdiff_mat[mask_mat==1]
        
        passive_mask = mask_mat.sum(axis=0)>0
        
        self.passiveloss_sum[passive_mask] = self.passiveloss_sum[passive_mask] + (self.C1*train_lossdiff[passive_mask])
        self.passiveloss_counts[passive_mask] = self.passiveloss_counts[passive_mask] + 1
        
        self.activeloss_sum[batch_indices] = self.activeloss_sum[batch_indices] + batch_lossdiff
        # print(np.mean(batch_lossdiff>=0))
        self.activeloss_counts[batch_indices] = self.activeloss_counts[batch_indices] + 1
        
        self.update_graph_mat_oneshot(
            batch_indices[locations_temp[:,0]],
            locations_temp[:,1],
            trainlossdiff_flat,
            class_normalizer
        )
        
        
    def store_graph(self,folder,loader_params,influence_params,train_params):
        self.update_normalized_graph()
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.chdir(folder)
        
        newpath = loader_params['dataset_name'] 
                           
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        os.chdir(newpath)
            
        files = os.listdir()
        
        ID = str(1+int(len(files)/2))
        sparse.save_npz(ID+".npz", self.normgraph_mat)
        with open(ID +'.pkl', 'wb') as handle:
            pickle.dump([loader_params, influence_params, train_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        os.chdir('..')
        os.chdir('..')
        
        return
    
    def load_graph(self,folder, dataset_name,ID):
        self.update_normalized_graph()
        os.chdir(folder)
        
        os.chdir(dataset_name)
        if ID == 'latest':
            files = os.listdir()
            ID = str(int(len(files)/2))
            print("File ID:", ID)
            
        self.normgraph_mat = sparse.load_npz(str(ID)+".npz")
        
        os.chdir('..')
        os.chdir('..')
        
        return self.normgraph_mat
    
    

class InfluenceGraphv3: # Gradient descent based approaches

    def __init__(self, node_size,node_labels,batch_update_size,influence_params):
        self.node_size = node_size
        self.node_labels = node_labels
        self.node_list = list(range(0,node_size))
        
        self.locations = np.array(list(itertools.product(np.arange(0,batch_update_size),self.node_list)))
        self.locations = self.locations.reshape((batch_update_size,self.node_size,2))
        
        self.class_normalize = influence_params['class_normalize']
        self.remove_negatives = influence_params['remove_negatives']
        self.clipping = influence_params['clipping']
        self.intraclass_only = influence_params['intraclass_only']
        self.negative_clipping = influence_params['negative_clipping']
        self.gradient_lr = influence_params['gradient_lr']
        self.mode = influence_params['mode'] 
        
        row = np.array([0])
        col = np.array([0])
        data = np.array([0],dtype = influence_params['dtype'])
        self.graph_mat = csr_matrix((data, (row, col)), shape = (node_size,node_size))
        self.graph_counts = csr_matrix((data, (row, col)), shape = (node_size,node_size))
        
    def update_normalized_graph(self):
        if self.mode == 'mean':
            self.normgraph_mat = self.graph_mat.multiply(self.graph_counts.power(-1.))
        elif self.mode == 'grad':
            self.normgraph_mat = self.graph_mat
            
        return self.normgraph_mat
    
        
    def update_graph_mat_oneshot(self, locations_x,locations_y, weights, class_normalizer = 1):
        
        #  flattening in row-major order: (iterate through all columns for each row)
        #  Assume weights are already flattened
        
        weight_mat = csr_matrix((weights, (locations_x,locations_y)), shape = (self.node_size,self.node_size))
        count_mat = csr_matrix((class_normalizer*np.ones(len(locations_x)), (locations_x,locations_y)), shape = (self.node_size,self.node_size))

        self.graph_mat  = self.graph_mat + weight_mat
        self.graph_counts = self.graph_counts + count_mat
        
        
    def update_influence_graph(self, batch_indices, batch_lossdiff, train_lossdiff):
        
        # Default params: class_normalize = False, remove_negatives = False, clipping = True, intraclass_only = True
        #  itertools also performs computations in row major order 
        batch_indices = batch_indices.cpu()
        epsilon = 0.00000001
        
        # locations = np.array(list(itertools.product(batch_indices,self.node_list)))
        # locations = locations.reshape((len(batch_indices),self.node_size,2))
        mask_mat = np.ones((len(batch_indices),self.node_size))
        
        mask_mat[:,batch_indices] = 0 
        
        if self.class_normalize:
            class_normalizer = len(batch_indices)
        else:
            class_normalizer = 1
            
        
        if self.intraclass_only:
            batchlabel_mat = np.tile(np.expand_dims(self.node_labels[batch_indices],1), (1, self.node_size))
            reflabel_mat = np.tile(self.node_labels, (len(batch_indices),1))
            labeleq_mat = batchlabel_mat == reflabel_mat
            mask_mat = mask_mat * labeleq_mat 

        
        if self.remove_negatives:
            train_lossdiff = train_lossdiff * (train_lossdiff>0)
        
        lossdiff_mat = np.tile(train_lossdiff, (len(batch_indices),1))
        
        # batchlossdiff_mat = np.tile(batch_lossdiff + epsilon,(1,self.node_size))
        batchlossdiff_mat = np.tile(np.expand_dims(batch_lossdiff,1),(1,self.node_size))
        
        if self.mode == 'grad':
            W = self.graph_mat[batch_indices,:].toarray()
            predicted_lossdiffs = np.mean(W*batchlossdiff_mat,axis=0)
            err_mat = lossdiff_mat - predicted_lossdiffs
            print(np.abs(err_mat).mean()/np.abs(lossdiff_mat).mean())
            
            weights_mat =  2*self.gradient_lr*err_mat*batchlossdiff_mat/float(len(batch_indices))
            
        elif self.mode =='mean':
            squared_lossdiff_batch = np.sum(batch_lossdiff**2)
            weights_mat = lossdiff_mat*batchlossdiff_mat /squared_lossdiff_batch
            
        
        if self.clipping:
            weights_mat[weights_mat>1] = 1
        
        if self.negative_clipping:
            weights_mat[weights_mat<-1] = -1
        
        
        locations_temp = self.locations[mask_mat==1,:]
        weights_flat = weights_mat[mask_mat==1]
        
        self.update_graph_mat_oneshot(batch_indices[locations_temp[:,0]],locations_temp[:,1],
                                      weights_flat, class_normalizer)
        
        
    def store_graph(self,folder,loader_params,influence_params,train_params):
        self.update_normalized_graph()
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.chdir(folder)
        
        newpath = loader_params['dataset_name'] 
                           
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        os.chdir(newpath)
            
        files = os.listdir()
        
        ID = str(1+int(len(files)/2))
        sparse.save_npz(ID+".npz", self.normgraph_mat)
        with open(ID +'.pkl', 'wb') as handle:
            pickle.dump([loader_params, influence_params, train_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        os.chdir('..')
        os.chdir('..')
        
        return
    
    def load_graph(self,folder, dataset_name,ID):
        self.update_normalized_graph()
        os.chdir(folder)
        
        os.chdir(dataset_name)
        if ID == 'latest':
            files = os.listdir()
            ID = str(int(len(files)/2))
            print("File ID:", ID)
            
        self.normgraph_mat = sparse.load_npz(str(ID)+".npz")
        
        os.chdir('..')
        os.chdir('..')
        
        return self.normgraph_mat
        
    
    
class InfluenceGraphv4: # 2nd order approaches (Correlation Based)

    def __init__(self, node_size,node_labels,batch_update_size, influence_params, transform_params=None):
        self.node_size = node_size
        self.node_labels = node_labels
        self.transform_params = transform_params
        self.node_list = list(range(0,node_size))
        
        self.locations = np.array(list(itertools.product(np.arange(0,batch_update_size),self.node_list)))
        self.locations = self.locations.reshape((batch_update_size,self.node_size,2))
        
        self.class_normalize = influence_params['class_normalize']
        self.remove_negatives = influence_params['remove_negatives']
        self.clipping = influence_params['clipping']
        self.intraclass_only = influence_params['intraclass_only']
        self.negative_clipping = influence_params['negative_clipping']
        
        row = np.array([0])
        col = np.array([0])
        data = np.array([0],dtype = influence_params['dtype'])
        self.lossmult_sum = csr_matrix((data, (row, col)), shape = (node_size, node_size))
        self.passive_sum = csr_matrix((data, (row, col)), shape = (node_size, node_size))
        self.passive_square_sum = csr_matrix((data, (row, col)), shape = (node_size, node_size))
        self.active_sum =  csr_matrix((data, (row, col)), shape = (node_size, node_size))
        self.active_square_sum = csr_matrix((data, (row, col)), shape = (node_size, node_size))

        self.lossmult_counts = csr_matrix((data, (row, col)), shape = (node_size,node_size))
        
        
    def update_normalized_graph(self):
    
        epsilon = 0.1
        self.normgraph_mat = self.lossmult_sum.multiply(self.lossmult_counts.power(-1.))
        self.passive_sum = self.passive_sum.multiply(self.lossmult_counts.power(-1.))
        self.passive_square_sum = self.passive_square_sum.multiply(self.lossmult_counts.power(-1.))
        
        self.active_sum = self.active_sum.multiply(self.lossmult_counts.power(-1.))
        self.active_square_sum = self.active_square_sum.multiply(self.lossmult_counts.power(-1.))
        
        passive_std_mat = self.passive_square_sum - (self.passive_sum.power(2.0))
        passive_std_mat = passive_std_mat.power(0.5)
        
        active_std_mat = self.active_square_sum - (self.active_sum.power(2.0))
        active_std_mat = active_std_mat.power(0.5)
        
        std_product_mat = active_std_mat.multiply(passive_std_mat)
        
        mean_product_mat = self.passive_sum.multiply(self.active_sum)
        
        self.normgraph_mat = self.normgraph_mat - mean_product_mat
        self.normgraph_mat = self.normgraph_mat.multiply(std_product_mat.power(-1.))
        
        x,y = self.normgraph_mat.nonzero()
        vals = self.normgraph_mat.data
        vals = vals/np.sqrt(1.01-(vals**2))
        self.normgraph_mat = csr_matrix((vals, (x, y)), shape = (self.node_size,self.node_size))

        return self.normgraph_mat

        
    def update_graph_mat_oneshot(self, locations_x,locations_y, lossmult, trainlossdiff_flat, batchlossdiff_flat, class_normalizer = 1):
        
        #  flattening in row-major order: (iterate through all columns for each row)
        #  Assume weights are already flattened
        
        lossmult_mat = csr_matrix((lossmult, (locations_x,locations_y)), shape = (self.node_size,self.node_size))
        trainlossdiff_graph = csr_matrix((trainlossdiff_flat, (locations_x,locations_y)), shape = (self.node_size,self.node_size))
        batchlossdiff_graph = csr_matrix((batchlossdiff_flat, (locations_x,locations_y)), shape = (self.node_size,self.node_size))
        
        trainlossdiff_squared_graph = csr_matrix((trainlossdiff_flat**2, (locations_x,locations_y)), shape = (self.node_size,self.node_size))
        batchlossdiff_squared_graph = csr_matrix((batchlossdiff_flat**2, (locations_x,locations_y)), shape = (self.node_size,self.node_size))

        count_mat = csr_matrix((class_normalizer*np.ones(len(locations_x)), (locations_x,locations_y)), shape = (self.node_size,self.node_size))

        self.lossmult_sum  = self.lossmult_sum + lossmult_mat
        self.passive_sum = self.passive_sum + trainlossdiff_graph
        self.active_sum = self.active_sum + batchlossdiff_graph
        
        self.passive_square_sum = self.passive_square_sum + trainlossdiff_squared_graph
        self.active_square_sum = self.active_square_sum + batchlossdiff_squared_graph
        
        self.lossmult_counts = self.lossmult_counts + count_mat
        
        
    def update_influence_graph(self, batch_indices, batch_lossdiff, train_lossdiff):
        
        # Default params: class_normalize = False, remove_negatives = False, clipping = True, intraclass_only = True
        #  itertools also performs computations in row major order 
        batch_indices = batch_indices.cpu()
        epsilon = 0.00000001
        
        # locations = np.array(list(itertools.product(batch_indices,self.node_list)))
        # locations = locations.reshape((len(batch_indices),self.node_size,2))
        mask_mat = np.ones((len(batch_indices),self.node_size))
        
        mask_mat[:,batch_indices] = 0 
        
        if self.class_normalize:
            class_normalizer = len(batch_indices)
        else:
            class_normalizer = 1
            
        
        if self.intraclass_only:
            batchlabel_mat = np.tile(np.expand_dims(self.node_labels[batch_indices],1), (1, self.node_size))
            reflabel_mat = np.tile(self.node_labels, (len(batch_indices),1))
            labeleq_mat = batchlabel_mat == reflabel_mat
            mask_mat = mask_mat * labeleq_mat 
        
            # old - new
        
        if self.remove_negatives:
            train_lossdiff = train_lossdiff * (train_lossdiff>0)
        
        trainlossdiff_mat = np.tile(train_lossdiff, (len(batch_indices),1))
        
        # batchlossdiff_mat = np.tile(batch_lossdiff,(1,self.node_size))
        batchlossdiff_mat = np.tile(np.expand_dims(batch_lossdiff,1),(1,self.node_size))

        
        if self.clipping:
            trainlossdiff_mat[trainlossdiff_mat>1] = 1
        
        if self.negative_clipping:
            trainlossdiff_mat[trainlossdiff_mat<-1] = -1
            
        trainlossmult_mat = trainlossdiff_mat*batchlossdiff_mat
        
        locations_temp = self.locations[mask_mat==1,:]
        trainlossmult_flat = trainlossmult_mat[mask_mat==1]
        trainlossdiff_flat = trainlossdiff_mat[mask_mat==1]
        batchlossdiff_flat = batchlossdiff_mat[mask_mat == 1]
        
        self.update_graph_mat_oneshot(
            batch_indices[locations_temp[:,0]],
            locations_temp[:,1],
            trainlossmult_flat,
            trainlossdiff_flat,
            batchlossdiff_flat,
            class_normalizer
        )
        
    
    def prune_graph(self, abs_threshold):
        
        x,y = self.normgraph_mat.nonzero()
        vals = self.normgraph_mat.data
        indices = np.abs(vals)>abs_threshold
        filtered_vals = vals[indices]
        filtered_x = x[indices]
        filtered_y = y[indices]
        
        self.normgraph_mat = csr_matrix(
            (filtered_vals, (filtered_x, filtered_y)),
            shape=(self.normgraph_mat.shape[0], self.normgraph_mat.shape[1])
        )
        return self.normgraph_mat

        
    def store_graph(self,folder,loader_params,influence_params,train_params):
        
        influence_params['graph_type'] = influence_params['graph_type'].__name__
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        os.chdir(folder)
        
        newpath = loader_params['dataset_name'] 
                           
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        os.chdir(newpath)
            
        files = os.listdir()
        
        ID = str(1+int(len(files)/2))
        sparse.save_npz(ID+".npz", self.normgraph_mat)
        with open(ID +'.pkl', 'wb') as handle:
            pickle.dump([loader_params, influence_params, train_params, self.node_labels,self.transform_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        os.chdir('..')
        os.chdir('..')
        
        return
    
    def load_graph(self,folder, dataset_name,ID):
        # self.update_normalized_graph()
        os.chdir(folder)
        
        os.chdir(dataset_name)
        if ID == 'latest':
            files = os.listdir()
            ID = str(int(len(files)/2))
            print("File ID:", ID)
            
        self.normgraph_mat = sparse.load_npz(str(ID)+".npz")
        
        os.chdir('..')
        os.chdir('..')
        
        return self.normgraph_mat
    
    


# class InfluenceGraphv4: # 2nd order approaches (Correlation Based)

    # def __init__(self, node_size,node_labels,batch_update_size, influence_params, transform_params=None):
    #     self.node_size = node_size
    #     self.node_labels = node_labels
    #     self.transform_params = transform_params
    #     self.node_list = list(range(0,node_size))
        
    #     self.locations = np.array(list(itertools.product(np.arange(0,batch_update_size),self.node_list)))
    #     self.locations = self.locations.reshape((batch_update_size,self.node_size,2))
        
    #     self.class_normalize = influence_params['class_normalize']
    #     self.remove_negatives = influence_params['remove_negatives']
    #     self.clipping = influence_params['clipping']
    #     self.intraclass_only = influence_params['intraclass_only']
    #     self.negative_clipping = influence_params['negative_clipping']
        
        
    #     row = np.array([0])
    #     col = np.array([0])
    #     data = np.array([0],dtype = influence_params['dtype'])
    #     self.lossmult_sum = csr_matrix((data, (row, col)), shape = (node_size,node_size))
    #     self.passive_sum = csr_matrix((data, (row, col)), shape = (node_size,node_size))
    #     self.passive_square_sum = csr_matrix((data, (row, col)), shape = (node_size,node_size))
    #     self.lossmult_counts = csr_matrix((data, (row, col)), shape = (node_size,node_size))
    #     self.passiveloss_sum  = np.zeros(self.node_size)
    #     self.passiveloss_counts  = np.zeros(self.node_size)
    #     self.passiveloss_square_sum = np.zeros(self.node_size)
    #     self.activeloss_square_sum = np.zeros(self.node_size)
    #     self.activeloss_sum  = np.zeros(self.node_size)
    #     self.activeloss_counts  = np.zeros(self.node_size)
        
        
    # def update_normalized_graph(self):
        
    #     activeloss_mean = self.activeloss_sum/self.activeloss_counts
    #     activeloss_square_mean = self.activeloss_square_sum/self.activeloss_counts
        
    #     activeloss_std = np.sqrt(activeloss_square_mean - (activeloss_mean**2))
    
    
    #     epsilon = 0.1
    #     self.normgraph_mat = self.lossmult_sum.multiply(self.lossmult_counts.power(-1.))
    #     self.passive_sum = self.passive_sum.multiply(self.lossmult_counts.power(-1.))
    #     self.passive_square_sum = self.passive_square_sum.multiply(self.lossmult_counts.power(-1.))
        
        
    #     rows,cols = self.normgraph_mat.nonzero()
        
    #     active_mean_mat = csr_matrix((activeloss_mean[rows], (rows,cols)), shape = (self.node_size,self.node_size))
    #     active_std_mat = csr_matrix((activeloss_std[rows], (rows,cols)), shape = (self.node_size,self.node_size))
        
    #     passive_std_mat = self.passive_square_sum - (self.passive_sum.power(2.0))
        
    #     std_product_mat = active_std_mat.multiply(passive_std_mat.power(0.5))
        
    #     mean_product_mat = self.passive_sum.multiply(active_mean_mat)
        
    #     self.normgraph_mat = self.normgraph_mat - mean_product_mat
    #     self.normgraph_mat = self.normgraph_mat.multiply(std_product_mat.power(-1.))
        
        
    #     return self.normgraph_mat

        
    # def update_graph_mat_oneshot(self, locations_x,locations_y, lossmult, trainlossdiff_flat, class_normalizer = 1):
        
    #     #  flattening in row-major order: (iterate through all columns for each row)
    #     #  Assume weights are already flattened
        
    #     lossmult_mat = csr_matrix((lossmult, (locations_x,locations_y)), shape = (self.node_size,self.node_size))
    #     trainlossdiff_graph = csr_matrix((trainlossdiff_flat, (locations_x,locations_y)), shape = (self.node_size,self.node_size))
        
        
    #     count_mat = csr_matrix((class_normalizer*np.ones(len(locations_x)), (locations_x,locations_y)), shape = (self.node_size,self.node_size))

    #     self.lossmult_sum  = self.lossmult_sum + lossmult_mat
    #     self.passive_sum = self.passive_sum + trainlossdiff_graph
    #     self.passive_square_sum = self.passive_square_sum + (trainlossdiff_graph.power(2.0))
    #     self.lossmult_counts = self.lossmult_counts + count_mat
        
        
        
    # def update_influence_graph(self, batch_indices, batch_lossdiff, train_lossdiff):
        
    #     # Default params: class_normalize = False, remove_negatives = False, clipping = True, intraclass_only = True
    #     #  itertools also performs computations in row major order 
    #     batch_indices = batch_indices.cpu()
    #     epsilon = 0.00000001
        
    #     # locations = np.array(list(itertools.product(batch_indices,self.node_list)))
    #     # locations = locations.reshape((len(batch_indices),self.node_size,2))
    #     mask_mat = np.ones((len(batch_indices),self.node_size))
        
    #     mask_mat[:,batch_indices] = 0 
        
    #     if self.class_normalize:
    #         class_normalizer = len(batch_indices)
    #     else:
    #         class_normalizer = 1
            
        
    #     if self.intraclass_only:
    #         batchlabel_mat = np.tile(np.expand_dims(self.node_labels[batch_indices],1), (1, self.node_size))
    #         reflabel_mat = np.tile(self.node_labels, (len(batch_indices),1))
    #         labeleq_mat = batchlabel_mat == reflabel_mat
    #         mask_mat = mask_mat * labeleq_mat 
        
    #         # old - new
        
        
    #     if self.remove_negatives:
    #         train_lossdiff = train_lossdiff * (train_lossdiff>0)
        
    #     trainlossdiff_mat = np.tile(train_lossdiff, (len(batch_indices),1))
        
    #     # batchlossdiff_mat = np.tile(batch_lossdiff,(1,self.node_size))
    #     batchlossdiff_mat = np.tile(np.expand_dims(batch_lossdiff,1),(1,self.node_size))
        
        
        
        
    #     if self.clipping:
    #         trainlossdiff_mat[trainlossdiff_mat>1] = 1
        
    #     if self.negative_clipping:
    #         trainlossdiff_mat[trainlossdiff_mat<-1] = -1
            
            
    #     trainlossmult_mat = trainlossdiff_mat*batchlossdiff_mat
        
        
    #     locations_temp = self.locations[mask_mat==1,:]
    #     trainlossmult_flat = trainlossmult_mat[mask_mat==1]
    #     trainlossdiff_flat = trainlossdiff_mat[mask_mat==1]
        
    #     self.activeloss_sum[batch_indices] = self.activeloss_sum[batch_indices] + batch_lossdiff
    #     self.activeloss_square_sum[batch_indices] = self.activeloss_square_sum[batch_indices] + (batch_lossdiff**2)
        
    #     # print(np.mean(batch_lossdiff>=0))
    #     self.activeloss_counts[batch_indices] = self.activeloss_counts[batch_indices] + 1
        
        
        
    #     self.update_graph_mat_oneshot(batch_indices[locations_temp[:,0]],locations_temp[:,1],
    #                                   trainlossmult_flat,trainlossdiff_flat, class_normalizer)
        
    
    # def prune_graph(self, abs_threshold):
        
    #     x,y = self.normgraph_mat.nonzero()
    #     vals = self.normgraph_mat.data
    #     indices = np.abs(vals)>abs_threshold
    #     filtered_vals = vals[indices]
    #     filtered_x = x[indices]
    #     filtered_y = y[indices]
        
    #     self.normgraph_mat = csr_matrix((filtered_vals, (filtered_x, filtered_y)), 
    #                                     shape = (self.normgraph_mat.shape[0],self.normgraph_mat.shape[1]))
        
    #     return self.normgraph_mat

        
    # def store_graph(self,folder,loader_params,influence_params,train_params):
        
    #     influence_params['graph_type'] = influence_params['graph_type'].__name__
        
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    #     os.chdir(folder)
        
        
    #     newpath = loader_params['dataset_name'] 
                           
    #     if not os.path.exists(newpath):
    #         os.makedirs(newpath)
    #     os.chdir(newpath)
            
    #     files = os.listdir()
        
    #     ID = str(1+int(len(files)/2))
    #     sparse.save_npz(ID+".npz", self.normgraph_mat)
    #     with open(ID +'.pkl', 'wb') as handle:
    #         pickle.dump([loader_params, influence_params, train_params, self.node_labels,self.transform_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    #     os.chdir('..')
    #     os.chdir('..')
        
    #     return
    
    # def load_graph(self,folder, dataset_name,ID):
    #     # self.update_normalized_graph()
    #     os.chdir(folder)
        
    #     os.chdir(dataset_name)
    #     if ID == 'latest':
    #         files = os.listdir()
    #         ID = str(int(len(files)/2))
    #         print("File ID:", ID)
            
    #     self.normgraph_mat = sparse.load_npz(str(ID)+".npz")
        
    #     os.chdir('..')
    #     os.chdir('..')
        
    #     return self.normgraph_mat
    
    
    
class IG_Measures: 

    def __init__(self, IG,node_labels):
        
        self.IG_network = nx.from_scipy_sparse_array(IG)
        self.IG_sparsemat = IG
        
    
    def mean_in_degree(self):
        
        self.IG_network.in_degree(weight='weight')
        
        in_degrees = dict(self.IG_network.in_degree(weight='weight'))

        mean_in_degree = np.mean(list(in_degrees.values()))
        
        return mean_in_degree
    
    def group_influence_estimation(self, node_labels, binary_group_id, intraclass_only = True): # 0 -> 1 influence
        
        if intraclass_only:
            
            labels_list = np.unique(node_labels)
            denominator_sum = 0 
            numerator_sum = 0 
            for L in labels_list:
                binary_filter_0 =  binary_group_id == 0 and node_labels == L 
                binary_filter_1 =  binary_group_id == 1 and node_labels == L 
                denominator_sum += np.sum(binary_filter_1)
                numerator_sum += self.IG_sparsemat[binary_filter_0][:,binary_filter_1].sum()
            
            return numerator_sum/denominator_sum
        
        else:
            return self.IG_sparsemat[binary_group_id == 0][:, binary_group_id == 1].sum()/np.sum(binary_group_id == 1)

    def mean_in_cluster_degree(self,percentile_thresholds = np.arange(20,50,0.5)):
        
        x,y = self.IG_sparsemat.nonzero()
        vals = self.IG_sparsemat.data
        clustering_ratio = []
        
        for p in percentile_thresholds:
            threshold_value = np.percentile(vals,p)
            indices = vals>threshold_value
            x_p = x[indices]
            y_p = y[indices]
            vals_p = vals[indices]
            # print((vals_p).mean())
            
            IG_sparsemat_p = csr_matrix((vals_p, (x_p, y_p)), shape = (self.IG_sparsemat.shape[0],self.IG_sparsemat.shape[1]))
            num_connect_p,connect_IDs = scipy.sparse.csgraph.connected_components(IG_sparsemat_p,connection='weak')
            print(num_connect_p)
            
            Intra_group_connect_sum = 0
            Inter_group_connect_sum = 0 
            Inter_count = 0 

            for i in range(num_connect_p):
                intra_group_count = np.sum(connect_IDs == i)
                if intra_group_count == 1:
                    Intra_group_connect_sum += 1 
                else:
                    Intra_group_connect_sum += self.IG_sparsemat[connect_IDs == i][:, connect_IDs == i].sum()/(intra_group_count*(intra_group_count-1))
                
                for j in range(num_connect_p):
                    if j !=i:
                        sum_check = self.IG_sparsemat[connect_IDs == i][:, connect_IDs == j].mean()
                        if sum_check !=0:
                            Inter_group_connect_sum += sum_check
                            Inter_count = Inter_count + 1
            
            Intra_group_connect_mean = Intra_group_connect_sum/num_connect_p
            print(Intra_group_connect_mean)
            if Inter_count == 0:
                clustering_ratio.append(1)
            else:
                Inter_group_connect_mean = Inter_group_connect_sum/Inter_count
                clustering_ratio.append(Inter_group_connect_mean/Intra_group_connect_mean)
        
        print(clustering_ratio)
        
        
            
            
        
        




# g = InfluenceGraph(100000,np.ones(100000),dtype = np.int32)
# t = time.time()
# rows = list(range(0,64))

# locations = np.array(list(itertools.product(rows,list(range(0,100000)))))

# g.update_graph_mat_oneshot(locations,np.ones((64,100000)).flatten())


# # for row in rows:
# #     g.update_graph_mat(row,np.arange(0,5000),np.arange(0,5000))
    

# print('Time Taken:', time.time()-t)


# A = sp.lil_matrix((4,5)) 
# 



