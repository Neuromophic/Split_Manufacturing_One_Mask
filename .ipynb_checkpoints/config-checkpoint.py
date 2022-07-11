import os
import torch
import numpy
import random

# device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cuda:0'
device = 'cpu'

# training
lr  = 0.1
slr = 0.1
patience = 1000

# cross dataset normaolization
normalization = True

# architecture for pNN
hidden_topology = [3,3]

# hyperparameter
alpha = 0.001
pnorm = 1

# measuring-aware hyperparameter
m = 0.3
T = 0.1

# printing technology
gmin = 0.01
gmax = 10.

# circuit
ACT_eta1 = 0.134
ACT_eta2 = 0.962
ACT_eta3 = 0.183
ACT_eta4 = 24.10

NEG_eta1 = 0.104
NEG_eta2 = 0.899
NEG_eta3 = 0.056
NEG_eta4 = 3.858

def SetSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    random.seed(seed)
    numpy.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)                  
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    

