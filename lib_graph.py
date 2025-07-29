# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:08:27 2025

@author: User
"""
import numpy as np
import itertools 
import os
import pickle
import time 
import networkx as nx
import scipy

from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components


class InfluenceGraphv5: # 2nd order approaches (Correlation Based)

    def __init__(self, node_size,node_labels,batch_update_size, influence_params, transform_params=None):
        self.node_size = node_size
        self.node_labels = node_labels
        self.transform_params = transform_params
        self.node_list = list(range(0,node_size))
        
        labels, inv = np.unique(node_labels, return_inverse=True)
        self.n_blocks    = len(labels)
        self.block_sizes = np.bincount(inv)
        self.offsets     = np.concatenate([[0], np.cumsum(self.block_sizes)])
        self.max_B       = self.block_sizes.max()
        
        # 2) Map each node → (block_id, local_index)
        self._block_id    = inv                                # shape (N,)
        self._local_index = np.empty_like(inv)
        self.inverse_lookup = np.full((self.n_blocks, self.max_B), -1, dtype=int)
        for b in range(self.n_blocks):
            mask = (inv == b)
            self._local_index[mask] = np.arange(self.block_sizes[b])
            self.inverse_lookup[b, np.arange(self.block_sizes[b])] = np.flatnonzero(mask)
        
        dtype = influence_params['dtype']
        
        # 3) Allocate padded 3D arrays for all stats
        shape = (self.n_blocks, self.max_B, self.max_B)
        self.lossmult_data    = np.zeros(shape, dtype=dtype)
        self.passive_data     = np.zeros(shape, dtype=dtype)
        self.active_data      = np.zeros(shape, dtype=dtype)
        self.passive_sq_data  = np.zeros(shape, dtype=dtype)
        self.active_sq_data   = np.zeros(shape, dtype=dtype)
        self.count_data       = np.zeros(shape, dtype=dtype)
        
        # 4) Placeholder for normalized blocks
        self.norm_data = np.zeros(shape, dtype=dtype)

        
    def update_normalized_graph(self):
        eps = 1e-12
        shape = (self.n_blocks, self.max_B, self.max_B)
        corr_data = np.zeros(shape, dtype=np.float32)  # Placeholder for computed correlations

        # Loop over each block
        for b in range(self.n_blocks):
            print(b)
            C = self.count_data[b].astype(float)
            if np.all(C == 0):
                continue  # Skip empty blocks

            L  = self.lossmult_data[b]
            P  = self.passive_data[b]
            P2 = self.passive_sq_data[b]
            A  = self.active_data[b]
            A2 = self.active_sq_data[b]

            invC = 1.0 / (2 * C + eps)
            N    = L * invC
            P    = P * invC
            P2   = P2 * invC
            A    = A * invC
            A2   = A2 * invC

            P_std = np.sqrt(np.maximum(P2 - P**2, eps))
            A_std = np.sqrt(np.maximum(A2 - A**2, eps))

            corr = (N - (P * A)) / (P_std * A_std)
            corr = np.clip(corr, -1, 1)
            corr_data[b] = corr / np.sqrt(1.01 - corr**2)

        # Gather non-zero entries
        block_id, local_x, local_y = np.nonzero(self.count_data)
        values = corr_data[block_id, local_x, local_y]

        # Recover global coordinates
        row = self.inverse_lookup[block_id, local_x]
        col = self.inverse_lookup[block_id, local_y]
        print(row.min(),row.max())
        print(col.min(),col.max())

        # Construct sparse matrix
        N = len(self._block_id)
        self.normgraph_mat = coo_matrix((values, (row, col)), shape=(N, N)).tocsr()
        return self.normgraph_mat

        
    def update_graph_mat_oneshot(self, loc_x, loc_y, lossmult, train_diff, batch_diff, class_norm = 1):
        #  flattening in row-major order: (iterate through all columns for each row)
        #  Assume weights are already flattened

        # t = time.time()
        
        b    = self._block_id[loc_x]
        lx   = self._local_index[loc_x]
        ly   = self._local_index[loc_y]
        
        # Six fast C‐loops
        np.add.at(self.lossmult_data,    (b, lx, ly), lossmult)
        np.add.at(self.passive_data,     (b, lx, ly), train_diff)
        np.add.at(self.active_data,      (b, lx, ly), batch_diff)
        np.add.at(self.passive_sq_data,  (b, lx, ly), train_diff**2)
        np.add.at(self.active_sq_data,   (b, lx, ly), batch_diff**2)
        np.add.at(self.count_data,       (b, lx, ly), class_norm)

        # print(time.time() - t)
        
         
    def update_influence_graph(self, batch_indices, batch_lossdiff, train_lossdiff):
        # Default params: class_normalize = False, remove_negatives = False, clipping = True, intraclass_only = True
        #  itertools also performs computations in row major order 
        batch_indices = batch_indices.cpu()
        num_nodes = self.node_size

        # Initialize mask
        mask_mat = np.ones((len(batch_indices), num_nodes), dtype=bool)
        mask_mat[:, batch_indices] = False  # Exclude self-connections
        
        # Apply intraclass masking
        # if self.intraclass_only:
        batch_labels = self.node_labels[batch_indices]
        node_labels = self.node_labels
        labeleq_mat = (batch_labels[:, None] == node_labels[None, :])
        mask_mat &= labeleq_mat

        # Flatten indices where mask == True
        row_idx, col_idx = np.where(mask_mat)
    
        # Calculate loss multipliers
        trainlossdiff_mat = train_lossdiff[col_idx]
        batchlossdiff_mat = batch_lossdiff[row_idx]

        lossmult_flat = trainlossdiff_mat * batchlossdiff_mat

        # Update influence graph
        self.update_graph_mat_oneshot(
            batch_indices[row_idx],
            col_idx,
            lossmult_flat,
            trainlossdiff_mat,
            batchlossdiff_mat,
            1
        )

        
    def prune_graph(self, abs_threshold):
        x, y = self.normgraph_mat.nonzero()
        vals = self.normgraph_mat.data
        indices = np.abs(vals)>abs_threshold
        filtered_vals = vals[indices]
        filtered_x = x[indices]
        filtered_y = y[indices]
        
        self.normgraph_mat = csr_matrix(
            (filtered_vals, (filtered_x, filtered_y)),
            shape = (self.normgraph_mat.shape[0], self.normgraph_mat.shape[1])
        )
        return self.normgraph_mat

        
    def store_graph(self,folder,loader_params,influence_params,train_params):
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
            pickle.dump([
                loader_params,
                influence_params,
                train_params,
                self.node_labels,self.transform_params
            ], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
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


