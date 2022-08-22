#!/usr/bin/env python

#SBATCH --job-name=SpNNHidden

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hzhao@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:1

# Import library
import importlib
import torch
import pickle
import os
import sys
sys.path.append('/pfs/data5/home/kit/tm/px3192/Split_Manufacturing_One_Mask/')
import matplotlib.pyplot as plt
import numpy as np
import pNN_Split as pnn
import random
import config
import evaluation as E
import training
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

seed = int(sys.argv[1])

# create file
topology_name = ''
for t in config.hidden_topology:
    topology_name += str(t)
whole_file_path = f'./result/super pNN hidden {topology_name}'
if not os.path.exists(whole_file_path):
    os.mkdir(whole_file_path)
if not os.path.exists(whole_file_path + '/model'):
    os.mkdir(whole_file_path + '/model')
    
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


# load normalization factors
acc_reference = np.loadtxt(f'./result/seperate pNN hidden {topology_name}/acc_factor.txt').flatten()
train_reference = np.loadtxt(f'./result/seperate pNN hidden {topology_name}/train_factor.txt').flatten()
valid_reference = np.loadtxt(f'./result/seperate pNN hidden {topology_name}/valid_factor.txt').flatten()

if not config.normalization:
    train_reference = np.ones(len(datasets))
    valid_reference = np.ones(len(datasets))

acc_factor = 1 / acc_reference
train_factor = 1 / (train_reference+0.01)
valid_factor = 1 / (valid_reference+0.01)

acc_factor = (acc_factor / acc_factor.shape[0]).tolist()
train_factor = (train_factor / np.sum(train_factor)).tolist()
valid_factor = (valid_factor / np.sum(valid_factor)).tolist()

# hyper parameter for penalty
alphas = np.zeros([102])
alphas[1:-1] = np.logspace(np.log(1e-5), np.log(1e5), 100, base=np.e)
alphas[-1] = 1e6
alphas = np.round(alphas, 5)

for alpha in alphas:
    
    # run over seeds
    # for seed in range(10):
    
    if os.path.isfile(f'{whole_file_path}/model/spnn_{alpha}_{seed}'):
        print(f'Super pNN {alpha} {seed} exists, skip this training.')
    else:
        print(f'Training with penalty:{alpha} seed:{seed}.')

        config.SetSeed(seed)

        # SuperPNN
        ## Define
        SuperPNN = pnn.SuperPNN(num_in, num_out, config.hidden_topology)
        SuperPNN.to(device)
        optimizer = torch.optim.Adam(SuperPNN.parameters(), lr=config.slr)

        ## Training
        SuperPNN, _, _, _, _ = training.train_spnn(SuperPNN,
                                                   X_trains, y_trains,
                                                   X_valids, y_valids,
                                                   optimizer, pnn.LOSSFUNCTION,
                                                   train_factor, valid_factor, acc_factor,
                                                   alpha, UUID=f'hidden_{alpha}_{seed}')

        torch.save(SuperPNN, f'{whole_file_path}/model/spnn_{alpha}_{seed}')