# Import library
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
device = config.device
# device = torch.device('cuda:0')
# device = 'cpu'

# Prepare data
## Datasets
datasets = os.listdir('./Datasets/datasets/')
datasets = [d for d in datasets if d.endswith('.p')]
datasets.sort()

## Load data
names   = []
num_in  = []
num_out = []
X_trains = []
y_trains = []
X_valids = []
y_valids = []
X_tests = []
y_tests = []

for dataset in datasets:
    datapath = os.path.join('./Datasets/datasets/' + dataset)
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    X_train    = data['X_train']
    y_train    = data['y_train']
    X_valid    = data['X_valid']
    y_valid    = data['y_valid']
    data_name  = data['name']

    N_class    = data['n_class']
    N_feature  = data['n_feature']
    N_train    = X_train.shape[0]
    N_valid    = X_valid.shape[0]
    
    print('Loading', data_name, N_feature, N_class, N_train, N_valid)
    
    names.append(data_name)
    num_in.append(N_feature)
    num_out.append(N_class)
    
    X_trains.append(X_train.to(device))
    y_trains.append(y_train.to(device))
    X_valids.append(X_valid.to(device))
    y_valids.append(y_valid.to(device))

print('Finish data loading.')


# with/without normalization
normalization = config.normalization

# load normalization factors
acc_reference = np.loadtxt('./result/seperate pNN/acc_factor.txt').flatten()
train_reference = np.loadtxt('./result/seperate pNN/train_factor.txt').flatten()
valid_reference = np.loadtxt('./result/seperate pNN/valid_factor.txt').flatten()

if not normalization:
    train_reference = np.ones(len(datasets))
    valid_reference = np.ones(len(datasets))

acc_factor = 1 / acc_reference
train_factor = 1 / (train_reference+0.01)
valid_factor = 1 / (valid_reference+0.01)

acc_factor = (acc_factor / acc_factor.shape[0]).tolist()
train_factor = (train_factor / np.sum(train_factor)).tolist()
valid_factor = (valid_factor / np.sum(valid_factor)).tolist()
    
# hyper parameter for learning rate
slr = config.slr

# hyper parameter for penalty
alphas = np.zeros(50)
alphas[1:] = np.logspace(np.log(1e-4),np.log(10),49, base=np.e)
for alpha in alphas:
    
    # run over seeds
    for seed in range(10):

        if os.path.isfile(f'./result/super pNN/model/spnn_{alpha}_{seed}'):
            print(f'Super pNN {alpha} {seed} exists, skip this training.')
        else:
            print(f'Training with penalty:{alpha} seed:{seed}.')

            config.SetSeed(seed)

            # SuperPNN
            ## Define
            SuperPNN = pnn.SuperPNN(num_in, num_out, config.hidden_topology)
            SuperPNN.to(device)
            optimizer = torch.optim.Adam(SuperPNN.parameters(), lr=slr)

            ## Training
            SuperPNN, _, _, _, _ = training.train_spnn(SuperPNN,
                                                       X_trains, y_trains,
                                                       X_valids, y_valids,
                                                       optimizer, pnn.LOSSFUNCTION,
                                                       train_factor, valid_factor, acc_factor,
                                                       config.alpha)

            torch.save(SuperPNN, f'./result/super pNN/model/spnn_{alpha}_{seed}')