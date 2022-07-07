import torch
import numpy as np

def Acc(prediction, y):
    act, idx = torch.max(prediction, dim=1)
    corrects = (y.view(-1) == idx)
    return corrects.float().sum().item() / y.numel()

def ACC(predictions, Ys, factors=0):
    accs = []
    if factors==0:
        for p, y in zip(predictions, Ys):
            acc = Acc(p, y)
            accs.append(acc)
    else:
        for p, y, factor in zip(predictions, Ys, factors):
            acc = Acc(p, y)/factor
            accs.append(acc)
    return np.mean(accs)

def BASIC(nn, x, y, *args, **kwargs):
    '''
    classic accuracy of NN
    :param nn: neural network
    :param x: testing examples
    :param y: testing labels
    :return: accuracy
    '''
    act, idx = torch.max(nn(x), dim=1)
    corrects = (y.view(-1) == idx)

    return corrects.float().sum().item() / y.numel()


def MAA(nn, x, y, sensing_margin=.01):
    '''
    measuring-aware accuracy, considering the measuring-margin
    and the output of wrong classes, i.e., the output of wrong
    classes must be negative, otherwise is the classification still wrong.
    :param nn: neural network
    :param x: testing examples
    :param y: testing labels
    :param sensing_margin: sensing margin
    :return: measuring-aware accuracy
    '''
    # find the 2 highest output in predictions
    act, idx = torch.topk(nn(x), k=2, dim=1)
    # if the class with the highest output is larger than sensing-margin (sensible)
    # and the 2nd highest value is negative
    # the of course the highest output refers to the right class
    # then the classification is right
    corrects = (act[:, 0] >= sensing_margin) & (act[:, 1] <= 0) & (y.view(-1) == idx[:, 0])

    return corrects.float().sum().item() / y.numel()


