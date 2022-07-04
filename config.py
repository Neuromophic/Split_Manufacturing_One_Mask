import os

# path for dataset
path = os.path.join(os.getcwd(), 'Datasets', 'dataset_processed')
files = os.listdir(path)
datasets = [k.replace('Dataset_', '').replace('.p', '') for k in files if k.startswith('Dataset_')]
datasets.sort()

# which dataset is selected
current_dataset = 5

# learning-rate
lr = 0.005

# random seed for data-split
data_split_seed = 0

# architecture for pNN
hidden_topology = [5,5]

# hyperparameter
alpha = 0.01

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
