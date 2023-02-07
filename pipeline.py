# %% [markdown]
# # Pipeline

# %%
# Imports

import torch

import networkx as nx

import numpy as np
import pandas as pd
import scipy.io
 
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import pickle

from tensorly.decomposition import tucker, constrained_parafac

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix, issparse

from SSLH_inference import *
from SSLH_utils import *

from tensorly.contrib.sparse import decomposition
import sparse

from pyod.models.lof import LOF

# %% [markdown]
# ### Load Network & Ego Nets

# %%
def load_network(path):
    
    try:
        data = scipy.io.loadmat(path)
    except:
        print('Invalid data path')

    G = nx.from_scipy_sparse_array(data["Network"])
    # nx.set_node_attributes(G, bc_data["Attributes"], 'Attributes')
    print(str(G))

    # convert list of lists to list
    labels = [j for i in data["Label"] for j in i]

    # Add labels to each node
    for i in range(len(G.nodes)):
        G.nodes[i]['Anomaly'] = labels[i]

    is_undirected = not nx.is_directed(G)

    # G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
    # G = nx.convert_node_labels_to_integers(G)

    ego_gs, roots = [], []

    # if 0-degree node(s), remove label(s) from consideration
    if len(labels) != G.number_of_nodes():
        labels = list(nx.get_node_attributes(G, 'Anomaly').values())

    for i in tqdm(range(G.number_of_nodes())):
        roots.append(G.nodes[i]['Anomaly'])
        G_ego = nx.ego_graph(G, i, radius=1, undirected=is_undirected)
        if G_ego.number_of_nodes() >= 2:
            ego_gs.append(G_ego)

    return G, ego_gs, roots, labels


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here

# %%
    load = input('Load previous loaded network? (y/n): ')
    if load.lower()[0] == 'n':

        dataset = int(input('Enter dataset/network path: \n\t (1) BlogCatalog \n\t (2) Flickr \n\t (3) ACM\n'))

        if dataset == 1: 
            data_path = 'datasets/blogcatalog.mat'
            dataset = 'bc'
        elif dataset == 2: 
            data_path = 'datasets/Flickr.mat'
            dataset = 'flr'
        elif dataset == 3: 
            data_path = 'datasets/ACM.mat'
            dataset = 'acm'
        
        G, ego_gs, roots, labels = load_network(data_path)

        path = f'{dataset}_data.sav'

        saved_model = open(path, 'wb')
        pickle.dump((G, ego_gs, roots, labels), saved_model)
        saved_model.close()

    else:
        data_path = input('Enter file path: ')
        saved_model = open(data_path , 'rb')
        G, ego_gs, roots, labels = pickle.load(saved_model)
        saved_model.close()
        roots = [int(r) for r in roots]
        dataset = data_path.split('_')[0]
        print()

# %%
    print(f'Using {len(ego_gs)} egonets')

# %% [markdown]
# ### Sparse Tensor Construction

    load = input('Load previous constructed tensor? (y/n): ')

    if load.lower()[0] == 'n':

    # %%
        N = G.number_of_nodes()

        # %%
        indices = []
        padded_gs = []

        undirected = not nx.is_directed(G)

        for i, g in enumerate(tqdm(ego_gs)):
            ego_adj_list = dict(g.adjacency())
            
            result = np.zeros((N, N))
            for node in ego_adj_list.keys():    
                for neighbor in ego_adj_list[node].keys():

                    result[node][neighbor] = 1
                    if undirected:
                        result[neighbor][node] = 1
                    indices.append([i, node, neighbor])
                    indices.append([i, neighbor, node])
                    
            padded_gs.append(result)

        # values, indices = [], []
        # padded_gs = []

        # undirected = not nx.is_directed(G)

        # for i, g in enumerate(tqdm(ego_gs)):
        #     ego_adj_list = dict(g.adjacency())
            
        #     result = np.zeros((N, N))
        #     for node in ego_adj_list.keys():    
        #         for neighbor in ego_adj_list[node].keys():

        #             result[node][neighbor] = 1
        #             if undirected:
        #                 result[neighbor][node] = 1
        #                 indices.append([i, node, neighbor])
        #                 indices.append([i, neighbor, node])
                    
        #     norm = np.linalg.norm(result, ord='fro')
        #     values.append((g.number_of_edges(), norm))
        #     padded_gs.append(result * (1/norm))

    # %%
        i = torch.tensor(list(zip(*indices)))
        values = torch.ones(len(indices))

        cube = sparse.COO(i, data=values)

        # cube_values = []
        # for num_edges, norm in values:
        #     norm_value = 1/norm
        #     for _ in range(num_edges):
        #         cube_values.append(norm_value)
        # ten_values = torch.tensor(cube_values)
        # cube = sparse.COO(i, data=ten_values)

        path = f'{dataset}_tensor.sav'

        saved_tensor = open(path, 'wb')
        pickle.dump(cube, saved_tensor)
        saved_tensor.close()

    else:
        data_path = input('Enter file path: ')
        saved_tensor = open(data_path , 'rb')
        cube = pickle.load(saved_tensor)
        saved_tensor.close()

# %% [markdown]
# ### Tensor Decomposition + Reconstruction Error

# %%
    ranks = [int(r) for r in input('Enter ranks, space separated: ').split()]

# %%
    scores = []
    for rank in ranks:
        print(f'\nUSING RANK {rank}\n')
        load = input('Load Reconstruction Errors? (y/n): ')
        # not loading previously calculated reconstruction errors
        if load.lower()[0] == 'n':

            # checking for valid input
            load = input('\nLoad Previous Decomposition? (y/n): ')
            while (load.lower()[0] != 'n' and load.lower()[0] != 'y'):
                print('Invalid Input!')
                load = input('Load Previous Decomposition? (y/n): ')
            decomp = input('Select Tucker (1) or CP (2) Decomposition: ')
            while (decomp != '1' and decomp != '2'):
                print('Invalid Input!')
                decomp = input('Select Tucker (1) or CP (2) Decomposition: ')

            decomp_alg = 'tkd' if decomp == '1' else 'cpd'    

            if load.lower()[0] == 'n':
                path = f'{dataset}_{decomp_alg}_r{rank}.sav'
                # path = input('Enter file name to save factors as: ')
                if decomp == '1':
                    print('Tucker Decomposition...')
                    _, factors = decomposition.tucker(cube, rank=rank, init='random')
                elif decomp == '2':
                    print('Parafac Decomposition...')
                    _, factors = decomposition.parafac(cube, rank=rank, init='random')
                print(f"Factors Saved to {path}\n")
                saved_model = open(path, 'wb')
                pickle.dump(factors, saved_model)
                saved_model.close()
            else:
                with open(input('Enter file path: '), 'rb') as f:
                    factors = pickle.load(f)
                    f.close()
                    print()
            
            A, B, C = factors
            if decomp == '1':
                A, B, C, = np.array(A), np.array(B), np.array(C)
            elif decomp == '2':
                A, B, C = A.todense(), B.todense(), C.todense()
                
            # path = input('Enter file name to save reconstruction errors: ')
            path = f'{dataset}_{decomp_alg}_r{rank}_errors.sav'

            errors = []
            print("Calculating Reconstruction Errors...")
            for gs in tqdm(padded_gs):
                if decomp == '1':
                    # projection
                    gs_p = ((A.T @ gs) @ B)
                    # reconstruction
                    gs_r = (A @ gs_p @ B.T)
                elif decomp == '2':
                    # projection
                    gs_p = ((np.linalg.pinv(A) @ gs) @ B)
                    # reconstruction
                    gs_r = (A @ gs_p @ np.linalg.pinv(B))
                d = np.linalg.norm(gs - gs_p, ord='fro')

                # # absolute error
                # errors.append(d / np.linalg.norm())

                # relative error
                errors.append(d / np.linalg.norm(gs, ord='fro'))

            errors = np.array(errors).reshape(-1, 1)

            saved_model = open(path, 'wb')
            pickle.dump(errors, saved_model)
            saved_model.close()
            print()

        # loading previously calculated reconstruction errors
        else:
            with open(input('Enter file path: '), 'rb') as f:
                errors = pickle.load(f)
                f.close()    
                print()    

        scale = MinMaxScaler()
        embeddings = scale.fit_transform(np.array(errors))

        scores.append(('No Model', rank, roc_auc_score(labels, embeddings)))

        ### pyOD Models

        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.33, random_state=42)

        clf = LOF()
        clf.fit(X_train, y_train)

        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the test data
        y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(X_test)  # outlier scores

        scores.append(('\nLOF', rank, roc_auc_score(y_test, y_test_scores)))


# %%
    for name, rank, auc in scores:
        print(f'Model: {name}, Rank: {rank}, AUC score: {auc}')


