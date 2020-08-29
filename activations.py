import torch
import nn

"""
This file contains the module classes that implement activation functions and non-linearity
in this small framework. Each activation function extends the base module and hence has a
forward and a backward method. The forward passes store the result, which is used in the next
backward call. The framework currently implements the following functions:

- Rectified Linear Unit (ReLU)
- Hyperbolic Tangent (Tanh)
- Logistic function (Sigmoid)


Authors: Albergoni, Bouvet, Feo
"""


"""
This module implements the Rectified Linear Unit activation function.
It stores the last activation computed with the forward method and uses
it upon calling the backward method.
"""
class ReLU(nn.Module):

    def __init__(self):
        super(ReLU, self).__init__()
        self._activation = None

    def forward(self, x):
        self._check.validate_input_type(x, 'torch.FloatTensor')
        
        self._activation = torch.clamp(x, min=0.0)
        return self._activation

    def backward(self, prevGradWrtoutput):
        self._check.validate_input_type(prevGradWrtoutput, 'torch.FloatTensor')
        self._check.shapes_match(self._activation, prevGradWrtoutput)

        derivative = self._activation.gt(0).type(torch.FloatTensor)
        return prevGradWrtoutput * derivative


"""
This module implements the Hyperbolic Tangent activation function.
It stores the last activation computed with the forward method and uses
upon calling the backward method.
"""
class Tanh(nn.Module):

    def __init__(self):
        super(Tanh, self).__init__()
        self._activation = None

    def forward(self, x):
        self._check.validate_input_type(x, 'torch.FloatTensor')
        
        self._activation = (x.exp() - (-x).exp())/(x.exp() + (-x).exp())
        return self._activation

    def backward(self, prevGradWrtoutput):
        self._check.validate_input_type(prevGradWrtoutput, 'torch.FloatTensor')
        self._check.shapes_match(self._activation, prevGradWrtoutput)

        tanh = (self._activation.exp() - (-self._activation).exp())/(self._activation.exp() + (-self._activation).exp())
        derivative = 1 - tanh**2
        return prevGradWrtoutput * derivative

    
"""
This module implements the Logistic activation function.
It stores the last activation computed with the forward method and uses
upon calling the backward method.
"""
class Sigmoid(nn.Module):
    
    def __init__(self):
        super(Sigmoid, self).__init__()
        self._activation = None

    def forward(self, x):
        self._check.validate_input_type(x, 'torch.FloatTensor')
        
        self._activation = x.exp()/(x.exp() + 1)
        return self._activation

    def backward(self, prevGradWrtoutput):
        self._check.validate_input_type(prevGradWrtoutput, 'torch.FloatTensor')
        self._check.shapes_match(self._activation, prevGradWrtoutput)
        
        sigma = prevGradWrtoutput.exp()/(prevGradWrtoutput.exp() + 1)
        return sigma * (1 - sigma)