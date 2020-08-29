import torch

from utils.helpers import TensorValidator
from utils.initialization import calculate_gain, init_xavier_normal

"""
This file contains the core modules of the small deep learning framework developed for
this mini-project. It contains the base module class that is then extended to implement
all the other building blocks, such as the Sequential and Linear modules that we can also
find in this file.

Authors: Albergoni, Bouvet, Feo
"""


""" 
Generic building block of the framework, can be extended to either activations, layers,
and loss functions and provides the skeleton for a backpropagation-compatible class with
the forward() and backward() methods.
"""
class Module(object):

    def __init__(self):
        self._check = TensorValidator()
        self.grad = {}
        self.params = {}

    def forward(self, input):
        raise NotImplementedError

    def backward(self, prevGradWrtoutput):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            self.grad[param] = torch.zeros(self.params[param].shape)


"""
This module serves as a container and manager for a sequence of other modules, and is the
top-level module of a net constructed with this framework. It is responsible of propagating
the forward/backward calls to all the encapsulated modules in the correct order.
"""
class Sequential(Module):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.layers = [arg for arg in args]

    def forward(self, x):
        tmp = x
        # Iterate over layers/components of the network
        for l in self.layers :
            tmp = l.forward(tmp) # note: intermediary outputs are saved locally by layers
        return tmp # only final output is returned

    def backward(self, prevGradWrtOutput):
        intermediaryGradient = prevGradWrtOutput
        # Iterate over layers/components of the network
        for l in reversed(self.layers) :
            intermediaryGradient = l.backward(intermediaryGradient)
        # Nothing to return, gradient is accumulated locally by layers
        
    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()


"""
This module implements a dense, fully connected layer. It stores the activations during the
forward call and uses them in the subsequent backward call.
It has weights and biases as parameters.
"""
class Linear(Module):

    def __init__(self, in_dim, out_dim, activation="linear", initialization="xavier"):
        super(Linear, self).__init__()
        
        # Initialization parameters
        self._activation = activation
        self._initialization = initialization
        self._in_dim = in_dim
        self._out_dim = out_dim

        # Init params and grad
        self._init_params()
        self.zero_grad()

        # Forward pass variables
        self._xin = None
        self._xout = None

    def forward(self, x):
        self._check.match(len(x.shape), 2)
        self._check.match(self._in_dim, x.shape[1])
        
        self._xin = x
        self._xout = torch.addmm(self.params['b'], x, self.params['w'].t())
        return self._xout

    def backward(self, prevGradWrtoutput):
        self._check.shapes_match(self._xout, prevGradWrtoutput)
            
        # Accumulate gradients
        self.grad['w'] += prevGradWrtoutput.t().mm(self._xin)/prevGradWrtoutput.shape[0]
        self.grad['b'] += prevGradWrtoutput.mean(0)
        
        # Return propagated gradients
        return prevGradWrtoutput.mm(self.params['w'])
    
    def _init_params(self):
        gain = calculate_gain(self._activation)
        if self._initialization == 'xavier':
            self.params['w'] = init_xavier_normal(self._in_dim, self._out_dim, gain=gain)
            self.params['b'] = torch.zeros(1, self._out_dim)
        else:
            raise NotImplementedError("This initialization scheme is not implemented")

