import torch
import math

"""
This file contains methods about parameter initializations.

Authors: Albergoni, Bouvet, Feo
"""


"""
Calculate the best gain value based on the activation function,
to be used in Xavier initialization.
"""
def calculate_gain(activation_function):

    if activation_function == 'linear':
        return 1
    elif activation_function == 'tanh':
        return 5.0 / 3
    elif activation_function == 'relu':
        return math.sqrt(2.0)
    else:
        raise NotImplementedError("This activation function has not been implemented")


"""
Initializes a 2D tensor of dimensions (fan_in, fan_out) with the normal
Xavier initialization scheme.
"""
def init_xavier_normal(fan_in, fan_out, gain=1.0):
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return torch.FloatTensor(fan_out, fan_in).normal_(0, std)   