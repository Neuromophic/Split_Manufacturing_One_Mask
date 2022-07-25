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
slr = 0.01
patience = 1000

# cross dataset normaolization
normalization = False

# architecture for pNN
MAX_IN = 9
MAX_out = 8
hidden = 2
hidden_topology = [hidden, hidden]
full_topology = [MAX_IN, hidden, MAX_out]
semi_topology = [hidden, MAX_out]

# hyperparameter
# alpha = 0.001
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
    
def DataReshape(Xs, num_in):
    if type(Xs) is list:
        Xs_new = []
        for x in Xs:
            diff = num_in - x.shape[1]
            if diff:
                x = torch.hstack([x, torch.zeros([x.shape[0], diff])])
                Xs_new.append(x)
            else:
                Xs_new.append(x)
        return Xs_new
    else:
        diff = num_in - Xs.shape[1]
        if diff:
            return torch.hstack([Xs, torch.zeros([Xs.shape[0], diff])])
        else:
            return Xs