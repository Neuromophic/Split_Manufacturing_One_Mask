import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


class PNNLayer(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(PNNLayer, self).__init__()
        # initialize theta
        # theta is [n_out , n_in + 2] vector. Plus two -> 1 for bias, 1 for decouple
        # a row of theta consists of [theta_1, theta_2, ..., theta_{n_in}, theta_b, theta_d]
        theta = torch.rand([n_in + 2, n_out])
        theta[-1,:] = theta[-1,:] * 100
        theta[-2,:] = 0.1788 / (1 - 0.1788) * (torch.sum(torch.abs(theta[:-3,:]), axis=0) + torch.abs(theta[-1,:]))
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)

    @property
    def theta(self):
        '''
        straight throgh for suitable theta range
        '''
        theta_temp = self.theta_.clone()
        theta_temp[theta_temp.abs()<0.01] = 0.
        theta_temp[theta_temp>1] = 1.
        theta_temp[theta_temp<-1] = -1.
        return theta_temp.detach() + self.theta_ - self.theta_.detach()

    @property
    def g(self):
        '''
        Get the absolute value of the surrogate conductance theta
        :return: absolute(theta)
        '''
        return self.theta.abs()
    
    @property
    def w(self):
        '''
        calculate weights from conductance
        '''
        return self.g / torch.sum(self.g, axis=0, keepdim=True)
    
    def inv(self, x):
        '''
        Quasi-negative value of x
        In w*x, when w < 0, it is not implementable as the resistor has no negative resistance.
        We conver the problem to (-w) * (-x) =: g * inv(x)
        :param x: values to be calculated
        :return: inv(x)
        '''
        # different constants for each columns later (variation)
        return 0.104 - 0.899 * torch.tanh((x + 0.056) * 3.858)

    def activate(self, z):
        '''
        activation function of PNN
        :param z: parameter after MAC
        :return: values after activation
        '''
        return 0.134 + 0.962 * torch.tanh((z - 0.183) * 24.10)

    def mac(self, a):
        '''
        calculate multiply-accumulate considering inv(a)
        :param a: input of the layer
        :return: output after MAC
        '''
        sign = self.theta.sign()
        positive = sign.clone()
        positive[positive<0] = 0
        negative = 1. - positive
        
        a_extend = torch.cat([a, torch.ones([a.shape[0],1]), torch.zeros([a.shape[0],1])], dim=1)
        a_neg = self.inv(a_extend)
        a_neg[:,-1] = 0.
        z = torch.matmul(a_extend, self.w*positive) + torch.matmul(a_neg, self.w*negative)
        return z

    def forward(self, a_previous):
        '''
        forward propagation: MAC and activation
        :param a: input of the layer
        :return: output of the layer
        '''
        z_new = self.mac(a_previous)
        a_new = self.activate(z_new)
        return a_new


# class PNNHiddenLayer(torch.nn.Module):
#     def __init__(self, n_in, n_out, super_theta):
#         super(PNNHiddenLayer, self).__init__()
#         theta = super_theta.view([n_in + 2, n_out])
#         theta[-1,:].data.copy_(theta[-1,:].data * 100)
#         theta[-2,:].data.copy_(0.1788/(1-0.1788)*(torch.sum(torch.abs(theta[:-3,:].data), dim=0)+torch.abs(theta[-1,:].data)))
#         self.theta_ = theta

#     @property
#     def theta(self):
#         '''
#         put theta into straight through function
#         '''
#         theta_temp = self.theta_.clone()
#         theta_temp[theta_temp.abs()<0.01] = 0.
#         theta_temp[theta_temp>1] = 1.
#         theta_temp[theta_temp<-1] = -1.
#         return theta_temp.detach() + self.theta_ - self.theta_.detach()

#     @property
#     def g(self):
#         '''
#         Get the absolute value of the surrogate conductance theta
#         :return: absolute(theta)
#         '''
#         return self.theta.abs()
    
#     @property
#     def w(self):
#         '''
#         calculate weights from conductance
#         '''
#         return self.g / torch.sum(self.g, axis=0, keepdim=True)
    
#     def inv(self, x):
#         '''
#         Quasi-negative value of x
#         In w*x, when w < 0, it is not implementable as the resistor has no negative resistance.
#         We conver the problem to (-w) * (-x) =: g * inv(x)
#         :param x: values to be calculated
#         :return: inv(x)
#         '''
#         # different constants for each columns later (variation)
#         return 0.104 - 0.899 * torch.tanh((x + 0.056) * 3.858)

#     def activate(self, z):
#         '''
#         activation function of PNN
#         :param z: parameter after MAC
#         :return: values after activation
#         '''
#         return 0.134 + 0.962 * torch.tanh((z - 0.183) * 24.10)

#     def mac(self, a):
#         '''
#         calculate multiply-accumulate considering inv(a)
#         :param a: input of the layer
#         :return: output after MAC
#         '''
#         sign = self.theta.sign()
#         positive = sign.clone()
#         positive[positive<0] = 0
#         negative = 1. - positive
        
#         a_extend = torch.cat([a, torch.ones([a.shape[0],1]), torch.zeros([a.shape[0],1])], dim=1)
#         a_neg = self.inv(a_extend)
#         a_neg[:,-1] = 0.
#         z = torch.matmul(a_extend, self.w*positive) + torch.matmul(a_neg, self.w*negative)
#         return z

#     def forward(self, a_previous):
#         '''
#         forward propagation: MAC and activation
#         :param a: input of the layer
#         :return: output of the layer
#         '''
#         z_new = self.mac(a_previous)
#         a_new = self.activate(z_new)
#         return a_new

    
class PNNHiddenLayer(PNNLayer):
    def __init__(self, n_in, n_out, super_theta):
        '''
        Generate a PNN layer
        :param n_in: number of input of the layer
        :param n_out: number of output of the layer
        '''
        super(PNNHiddenLayer, self).__init__(n_in, n_out)
        
        theta = super_theta.view([n_in + 2, n_out])
        theta[-1,:].data.copy_(theta[-1,:].data * 100)
        theta[-2,:].data.copy_(0.1788/(1-0.1788)*(torch.sum(torch.abs(theta[:-3,:].data), dim=0)+torch.abs(theta[-1,:].data)))
        self.theta_ = torch.nn.Parameter(theta, requires_grad=True)


class PNNNet(torch.nn.Module):
    def __init__(self, n_in, n_out, topology_hidden, hidden_theta):
        super(PNNNet, self).__init__()
        
        self.model = torch.nn.Sequential()
        
        # add input layer
        self.model.add_module(f'Input_Layer', PNNLayer(n_in, topology_hidden[0]))    
        
        # calculate hidden theta allocation
        split_position = [0]
        for l in range(len(topology_hidden)-1):
            split_position.append(split_position[-1] + (topology_hidden[l]+2)*topology_hidden[l+1])
        # add hidden layer
        for l in range(len(topology_hidden)-1):
            self.model.add_module(f'Hiddel_Layer {l}',
                                  PNNHiddenLayer(topology_hidden[l],
                                                 topology_hidden[l+1],
                                                 hidden_theta[split_position[l]:split_position[l+1]]))
        # add input layer
        self.model.add_module(f'Output_Layer', PNNLayer(topology_hidden[-1], n_out))    
        
    def forward(self, x):
        return self.model(x)       
        
        
        
class PNNSupernet(torch.nn.Module):
    def __init__(self, num_in_layers, num_out_layers, topology_hiddens):
        super(PNNSupernet, self).__init__()
        
        # topologies
        self.num_in   = num_in_layers
        self.topology = topology_hiddens
        self.num_out  = num_out_layers
        
        # count number of tasks
        self.N_tasks = len(num_in_layers)
        
        # count number of hidden theta (resistors)
        self.N_hidden_theta = 0
        for l in range(len(topology_hiddens)-1):
            self.N_hidden_theta += (topology_hiddens[l]+2)*topology_hiddens[l+1]
            
        # generate hidden theta for tasks   
        self.hidden_theta = torch.nn.Parameter(torch.rand([self.N_hidden_theta, self.N_tasks]), requires_grad=True)
        
        self.models = self.Build_pNNs()
        
    def Build_pNNs(self):
        models = []
        for n in range(self.N_tasks):
            models.append(PNNNet(self.num_in[n], self.num_out[n], self.topology, self.hidden_theta[:,n]))
        return torch.nn.ModuleList(models)


    def forward(self, Xs):
        y = []
        for nn, x in zip(self.models, Xs):
            y.append(nn(x))
        return y
    

def LossFunction(prediction, label, m=0.3, T=0.1):
    label = label.reshape(-1, 1)
    fy = prediction.gather(1, label).reshape(-1, 1)
    fny = prediction.clone()
    fny = fny.scatter_(1, label, -10 ** 10)
    fnym = torch.max(fny, axis=1).values.reshape(-1, 1)
    l = torch.max(m + T - fy, torch.tensor(0)) + torch.max(m + fnym, torch.tensor(0))
    L = torch.mean(l)
    return L

def LOSSFUNCTION(predictions, labels):
    L = 0
    for prediction, label in zip(predictions, labels):
        L = L + LossFunction(prediction, label)
    return L
