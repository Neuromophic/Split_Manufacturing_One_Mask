import importlib
import torch
import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pNN_Split as pnn
import random
import config
import evaluation as E
import training
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# Device
device = 'cpu'

# Prepare data
datasets = os.listdir('./Datasets/datasets/')
datasets = [d for d in datasets if d.endswith('.p')]
datasets.sort()

results = open('./result/seperate pNN/summary.txt', 'w')
results.close()

for dataset in datasets:
    train_loss_cross_seed = []
    valid_loss_cross_seed = []
    test_acc_cross_seed   = []
    test_maa_cross_seed   = []
    
    for seed in range(10):

        # Load data
        datapath = os.path.join('./Datasets/datasets/' + dataset)
        with open(datapath, 'rb') as f:
            data = pickle.load(f)

        X_train    = data['X_train']
        y_train    = data['y_train']
        X_valid    = data['X_valid']
        y_valid    = data['y_valid']
        X_test     = data['X_test']
        y_test     = data['y_test']
        data_name  = data['name']

        N_class    = data['n_class']
        N_feature  = data['n_feature']
        N_train    = X_train.shape[0]
        N_valid    = X_valid.shape[0]
        N_test     = X_test.shape[0]

        print(f'\n\nProcessing dataset: {data_name}\nN_train: {N_train}, N_valid: {N_valid}, N_test: {N_test}\nN_feature: {N_feature}, N_class: {N_class}\nseed: {seed}')
        
        # set seed
        random.seed(seed);
        np.random.seed(seed);
        torch.manual_seed(seed);

        # define pNN
        pNN = torch.nn.Sequential()
        pNN.add_module('Input Layer', pnn.pLayer(N_feature, config.hidden_topology[0]))
        for l in range(len(config.hidden_topology)-1):
            pNN.add_module(f'Hidden Layer {l}', pnn.pLayer(config.hidden_topology[l],config.hidden_topology[l+1]))
        pNN.add_module('Output Layer', pnn.pLayer(config.hidden_topology[-1],N_class))
        
        # define optimizer
        optimizer = torch.optim.Adam(pNN.parameters(), lr=config.lr)

        pNN, train_losses, valid_losses, _, _ = training.train_nn(pNN,
                                                DataLoader(TensorDataset(X_train,y_train), batch_size=N_train),
                                                DataLoader(TensorDataset(X_valid,y_valid), batch_size=N_valid),
                                                optimizer, pnn.LossFunction, device)
        
        torch.save(pNN, f'./result/seperate pNN/model/{data_name}_{seed}')
        
        test_acc_cross_seed.append(E.BASIC(pNN, X_test, y_test))
        test_maa_cross_seed.append(E.MAA(pNN, X_test, y_test))
        valid_loss_cross_seed.append(min(valid_losses))
        train_loss_cross_seed.append(min(train_losses))
    
    np.savetxt(f'./result/seperate pNN/results/{data_name}_test_acc.txt',
               np.array(test_acc_cross_seed).reshape(-1,1),     fmt='%.5f')
    np.savetxt(f'./result/seperate pNN/results/{data_name}_test_maa.txt',
               np.array(test_maa_cross_seed).reshape(-1,1),     fmt='%.5f')
    np.savetxt(f'./result/seperate pNN/results/{data_name}_valid_loss.txt',
               np.array(valid_loss_cross_seed).reshape(-1,1),   fmt='%.5f')
    np.savetxt(f'./result/seperate pNN/results/{data_name}_train_loss.txt',
               np.array(train_loss_cross_seed).reshape(-1,1),   fmt='%.5f')
    
    test_acc_mean   = np.mean(test_acc_cross_seed)
    test_acc_std    = np.std(test_acc_cross_seed)
    test_maa_mean   = np.mean(test_maa_cross_seed)
    test_maa_std    = np.std(test_maa_cross_seed)
    valid_loss_mean = np.mean(valid_loss_cross_seed)
    valid_loss_std  = np.std(valid_loss_cross_seed)
    train_loss_mean = np.mean(train_loss_cross_seed)
    train_loss_std  = np.std(train_loss_cross_seed)
    
    results = open('./result/seperate pNN/summary.txt', 'a')
    aligned_name = data_name.ljust(25,' ')
    results.write(f'{aligned_name}:\t\ttest acc: {test_acc_mean:.4f} ± {test_acc_std:.4f}\t\ttest maa: {test_maa_mean:.4f} ± {test_maa_std:.4f}\t\ttrain loss: {train_loss_mean:.4f} ± {train_loss_std:.4f}\t\tvalid loss: {valid_loss_mean:.4f} ± {valid_loss_std:.4f}\n')
    results.close()

        
        
        
        
        
        
        
        
        
        