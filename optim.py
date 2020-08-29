import torch

"""
This file contains the module classes that implement optimizers
in this small framework. Each optimizer extends the base module and hence has in
principle only a step() method to update parameters of the model that is being optimized.
The framework currently implements the following optimizers:

- Stochastic Gradient Descent

Authors: Albergoni, Bouvet, Feo
"""


"""
Base optimizer class, enforces the presence of the step() method.
"""
class Optimizer(object): 
    
    def __init__(self): 
        None
    
    def step(self): 
        raise NotImplementedError

        
"""
This module implements the Stochastic Gradient Descent optimizer.
In its step method, it updates each parameter of the model to be optimized
by taking a step in the direction of steepest descent according to a given
learning rate and to the gradients stored in the layers of the model.
"""
class SGD(Optimizer):
    
    def __init__(self, lr, model): 
        super(SGD, self).__init__()
        self.lr = lr
        self.model = model

    def step(self): 
        for layer in self.model.layers :
            for pname in layer.params : 
                layer.params[pname] -= (self.lr * layer.grad[pname])

