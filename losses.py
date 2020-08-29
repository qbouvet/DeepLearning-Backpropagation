import torch
import nn

"""
This file contains the module classes that implement loss functions
in this small framework. Each loss function extends the base module and hence has
a forward() and backward() method. The following loss functions are implemented
currently in the framework:

- Mean Squared Error (MSE)

Authors: Albergoni, Bouvet, Feo
"""

"""
This module implements the Mean Squared Error (MSE) loss function. It's forward method
can compute the loss on batches of predictions compared to true targets, aggregating the result
either by summing or averaging. The backward method instead computes the gradient of the
loss function wrt to predictions.
"""
class LossMSE(nn.Module):
    
    def __init__(self):
        super(LossMSE, self).__init__()
        self._supported_reductions = ['sum', 'mean']
        
    def _check_reduction(self, reduction):
        if reduction not in self._supported_reductions:
            raise ValueError("This loss reduction is not supported : %s" % reduction)
        
    def forward(self, x, t, reduction='sum'):
        self._check.shapes_match(x, t)
        self._check_reduction(reduction)
        
        if reduction == 'sum':
            return (x - t).pow(2).sum()
        elif reduction == 'mean':
            return (x - t).pow(2).sum(1).mean() 
        
    def backward(self, x, t):
        self._check.shapes_match(x, t)
        return 2 * (x - t)
