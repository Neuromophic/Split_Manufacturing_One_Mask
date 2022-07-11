import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import config


class pLayer(torch.nn.Module):
    def __init__(self, n_in, n_out):
        '''
        initialization of printed Layer (pLayer)
        :param n_in: input number of layer
        :param n_out: output number of layer
        '''
        super(pLayer, self).__init__()
        # surrogate conductance initialization
        theta = torch.rand([n_in + 2, n_out])/100. + config.gmin
        theta[-1, :] = theta[-1, :] + config.gmax
        theta[-2, :] = config.ACT_eta3 / (1. - config.ACT_eta3) * (torch.sum(theta[:-2, :], axis=0) + theta[-1, :])
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

    @property
    def theta(self):
        '''
        straight through (ST) for self.theta_
        :return: output adapts the manufacture technology
        '''
        # climp values outside [-gmax, gmax]
        self.theta_.data.clamp_(-config.gmax, config.gmax)
        # modify values inside [-gmin, gmin]
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs() < config.gmin] = 0.
        
        return theta_temp.detach() + self.theta_ - self.theta_.detach()

    @property
    def g(self):
        '''
        printed conductance, absolute value of surrogate theta
        '''
        return self.theta.abs()

    @property
    def w(self):
        '''
        convert printed conductance to corresponding weights in Neural network
        '''
        return self.g / torch.sum(self.g, axis=0, keepdim=True)

    def inv(self, x):
        '''
        if a surrogate conductance is negative, the corresponding weight should be negative,
        however, a negative conductance can not be produced.
        Therefore, we still print positive conductance,
        but convert the input voltage to negative.
        :param x: input of this printed layer (voltage)
        :return: neg(x), the quasi-negative value of x
        '''
        return config.NEG_eta1 - config.NEG_eta2 * torch.tanh((x + config.NEG_eta3) * config.NEG_eta4)

    def mac(self, a):
        '''
        calculate the weighted-sum of input data and weights.
        the input data corresponding to negative surrogate conductance will be transformed
        to the negative value, meanwhile, the positive conductance will be printed.
        :param a: input data of the pLayer (voltage)
        :return: weighted-sum
        '''

        # extract sign of surrogate conductance for the further processing of negative input
        sign = self.theta.sign()
        positive = sign.clone()
        positive[positive < 0] = 0
        negative = 1. - positive

        # extended input data with a "1" and a "0" corresponding to g_b and g_d
        a_extend = torch.cat([a,
                              torch.ones([a.shape[0], 1]).to(config.device),
                              torch.zeros([a.shape[0], 1]).to(config.device)], dim=1)
        a_neg = self.inv(a_extend)
        # negative extended data corresponding to g_d should still be 0
        a_neg[:, -1] = 0.

        # calculate weighted-sum
        z = torch.matmul(a_extend, self.w * positive) + torch.matmul(a_neg, self.w * negative)
        return z

    def activate(self, z):
        '''
        a tanh-like activation function implemented by hardware
        :param z: weighted-sum
        :return: activated value (voltage)
        '''
        return config.ACT_eta1 + config.ACT_eta2 * torch.tanh((z - config.ACT_eta3) * config.ACT_eta4)

    def forward(self, a_previous):
        '''
        weighted-sum followed by activation
        :param a_previous: input data of this layer
        :return:
        '''
        z_new = self.mac(a_previous)
        a_new = self.activate(z_new)
        return a_new


class pHiddenLayer(pLayer):
    '''
    printed hidden layer is exactly the same with printed layer exclusive the surrogate conductance.
    to achieve split manufacturing, all pNNs share the same hidden structure and hidden surrogate conductance.
    thus, the initial theta_ will be delivered from their supernet
    '''

    def __init__(self, n_in, n_out, super_theta):
        '''
        :param n_in: number of input
        :param n_out: number of output
        :param super_theta: common surrogate from supernet
        '''
        super(pHiddenLayer, self).__init__(n_in, n_out)
        # delete the self.theta_ initialized from parent class "pLayer"
        del self._parameters['theta_']
        # the theta in hidden layer consists of the common value from supernet for high-volumn printing
        # and an individual value for post-processing (conductance-reprinting)
        self.theta_common_ = super_theta
        self.theta_individual_ = torch.nn.Parameter(torch.rand([n_in + 2, n_out]) / 100., requires_grad=True)
    
    @property
    def theta_common(self):
        '''
        same as pLayer, straight through (ST) for self.theta_common_
        :return: output adapts the manufacture technology
        '''
        # clamp values outside [-gmax, gmax]
        self.theta_common_.data.clamp_(-config.gmax, config.gmax)
        # modify values inside [-gmin, gmin]
        theta_temp = self.theta_common_.clone()
        theta_temp[theta_temp.abs() < config.gmin] = 0.
        
        return theta_temp.detach() + self.theta_common_ - self.theta_common_.detach()
    
    @property
    def theta_individual(self):
        '''
        modify individual theta:
          1. theta_individual must be positive
          2. theta_individual must suitable for manufacturing
        :return: positive theta_individual that adapts the manufacture technology
        '''
        # keep theta_individual positive
        theta_individual_pos = self.theta_individual_.abs()
        
        theta_individual_pos.data.clamp_(0, config.gmax)
        # modify values inside [0, gmin]
        theta_temp = theta_individual_pos.clone()
        theta_temp[theta_temp < config.gmin] = 0.
        
        return theta_temp.detach() + theta_individual_pos - theta_individual_pos.detach()
    
    @property
    def theta(self):
        '''
        the resulted surrogate conductance, consisting of theta_common and theta_individual.
        since the reprinting can only increase the absolute value of conductance, i.e.,
        theta_individual in crease the theta_common. the value of theta_individual should have
        the same sign as theta_common, such that theta_common+theta_individual is larger
        in terms of absolute value
        :return: combination of common and individual theta
        '''
        return self.theta_common + torch.sign(self.theta_common)*self.theta_individual
    
    def forward(self, a_previous):
        '''
        weighted-sum followed by activation
        :param a_previous: input data of this layer
        :return:
        '''
        z_new = self.mac(a_previous)
        a_new = self.activate(z_new)
        return a_new
    

class pNN(torch.nn.Module):
    '''
    a pNN consists of 2 pLayers (1 for input layer, 1 for output layer) and multiple pHiddenLayers.
    the theta for pHiddenLayer is transferred from supernet
    '''
    def __init__(self, n_in, n_out, topology_hidden, hidden_theta):
        '''
        :param n_in: number of input
        :param n_out: number of output
        :param topology_hidden: a list, contains the structure of hidden layers
        :param hidden_theta: a list, contains commen theta for each hidden layer
        '''
        super(pNN, self).__init__()
        self.model = torch.nn.Sequential()

        # append input layer
        self.model.add_module(f'Input_Layer', pLayer(n_in, topology_hidden[0]))

        # append hidden layers and allocate common theta
        for l in range(len(topology_hidden) - 1):
            self.model.add_module(f'Hiddel_Layer {l}',
                                  pHiddenLayer(topology_hidden[l],
                                               topology_hidden[l + 1],
                                               hidden_theta[l]))

        # append output layer
        self.model.add_module(f'Output_Layer', pLayer(topology_hidden[-1], n_out))

    def forward(self, x):
        return self.model(x)


class SuperPNN(torch.nn.Module):
    '''
    super pNN contains multiple pNNs (tasks)
    '''
    def __init__(self, num_in_layers, num_out_layers, topology_hiddens):
        '''
        :param num_in_layers: a list, contains number of inputs for each task (PNN)
        :param num_out_layers:  a list, contains number of outputs for each task (PNN)
        :param topology_hiddens: a list, contains structure of hidden layers
        '''
        super(SuperPNN, self).__init__()
        self.num_in = num_in_layers
        self.topology = topology_hiddens
        self.num_out = num_out_layers

        # count number of tasks
        self.N_tasks = len(num_in_layers)

        # generate theta_common for hidden layers in pNNs
        self.hidden_theta_commen_ = []
        for l in range(len(topology_hiddens) - 1):
            # value initialization
            theta_temp = torch.rand([topology_hiddens[l] + 2, topology_hiddens[l + 1]]) / 100. + config.gmin
            theta_temp[-1, :] = theta_temp[-1, :] + config.gmax
            theta_temp[-2, :] = config.ACT_eta3 / (1 - config.ACT_eta3) * (torch.sum(theta_temp[:-2, :], dim=0) + theta_temp[-1, :])
            # assign name to common theta
            self.register_parameter(f'theta_common_ {l}', torch.nn.Parameter(theta_temp, requires_grad=True))
            # add to a list
            self.hidden_theta_commen_.append(self._parameters[f'theta_common_ {l}'])

        # build pNNs
        self.models = self.Build_pNNs()

    def Build_pNNs(self):
        '''
        build pNNs and collect them into a torch.nn.ModuleList
        :return: a torch.nn.ModuleList contains all pNNs (for all tasks)
        '''
        models = []
        for n in range(self.N_tasks):
            models.append(pNN(self.num_in[n], self.num_out[n], self.topology, self.hidden_theta_commen_))
        return torch.nn.ModuleList(models)

    def forward(self, Xs):
        '''
        calculate outputs for all pNNs in supernet and collect them in a list
        :param Xs: a list of training data for different tasks
        :return: a list of output for different tasks
        '''
        y = []
        for nn, x in zip(self.models, Xs):
            y.append(nn(x))
        return y
    
    def GetNorm(self, p=1):
        N = torch.tensor(0)
        M = 0
        for name,param in self.models.named_parameters():
            if name.endswith('theta_individual_'):
                N = N + param.norm(p)
                M = M + param.numel()
        return N / M


def LossFunction(prediction, label, m=0.3, T=0.1):
    '''
    Hardware-aware loss function for printed neuromorphic circuits
    :param prediction: output of pNN
    :param label: target (or groundtruth, or label)
    :param m: hyperparameter "margin"
    :param T: hyperparameter "measure resolution"
    :return: loss
    '''
    label = label.reshape(-1, 1)
    fy = prediction.gather(1, label).reshape(-1, 1)
    fny = prediction.clone()
    fny = fny.scatter_(1, label, -10 ** 10)
    fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
    l = torch.max(m + T - fy, torch.tensor(0)) + torch.max(m + fnym, torch.tensor(0))
    L = torch.mean(l)
    return L


def LOSSFUNCTION(predictions, labels, factors):
    '''
    Hardware-aware loss function for list
    :param predictions: a list of outputs from different pNNs
    :param labels: a list of targets/labels of different tasks
    :return: sum loss on all tasks
    '''
    L = 0
    for prediction, label, factor in zip(predictions, labels, factors):
        L = L + LossFunction(prediction, label) / (factor+torch.tensor(0.01))
    return L
