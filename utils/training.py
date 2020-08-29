import torch

"""
This file contains utility methods used to train the framework/pytorch modules and evaluate their performances.

Authors: Albergoni, Bouvet, Feo
"""


"""
This method is responsible for training a model constructed with the mini-framework
or a pure pytorch module.
It takes all the training parameters and objects and performs mini-batch optimization
over a given number of epochs. Records training and validation errors during the training
procedure in order to be able to inspect the evolution during and after training.

Parameters
----------
model: framework model
    model to be trained
    
train_input, train_target, val_input, val_target: 2D pytorch tensors
    training and validation sets
    
criterion: loss functionn
    loss function to optimize
    
opt: optimizer
    optimizer to be used at each training step, wraps the learning rate
    
epochs: int
    number of training epochs
    
mini_batch_size: int
    size of training batches
    
pytorch: boolean, default False
    specifies whether the model is a mini-framework or pytorch one
    
verbose: boolean, default False
    whether to print loss evolution during training
----------

Returns
----------
history: dict
    python dictionnary that contains the evolution of training and validation errors
----------
"""
def train_model(model,
                train_input, train_target, val_input, val_target,
                criterion, opt, epochs, mini_batch_size,
                pytorch=False, verbose=False):
    
    # History init
    history = {'train_loss': [], 'val_loss': []}
    
    # Iterate over epochs
    for e in range(epochs+1):
        
        # Loss accumulator for each epoch
        sum_loss = 0
        
        # Iterate over minibatches
        for b in range(0, train_input.size(0), mini_batch_size):
            
            # Get minibatches
            train_batch = train_input.narrow(0, b, mini_batch_size)
            target_batch = train_target.narrow(0, b, mini_batch_size)
            
            # Forward pass
            if not pytorch:
                train_pred = model.forward(train_batch)
                loss = criterion.forward(train_pred, target_batch)
            else:
                train_pred = model(train_batch)
                loss = criterion(train_pred, target_batch)
                
            sum_loss += loss.item()
                
            # Backward propagation
            model.zero_grad()
            
            if not pytorch:
                dl_dxout = criterion.backward(train_pred, target_batch)
                model.backward(dl_dxout)
            else:
                loss.backward()
        
            # Optimization step
            opt.step()
            
        # Update history
        if not pytorch:
            val_pred = model.forward(val_input)
            val_loss = criterion.forward(val_pred, val_target).item()
        else:
            val_pred = model(val_input)
            val_loss = criterion(val_pred, val_target).item()
        history['train_loss'].append(sum_loss)
        history['val_loss'].append(val_loss)
        
        # Log loss
        if verbose and e % 100 == 0:
            print("\tEpoch %s \t train_loss = %s \t val_loss = %s" % 
                  (e, '{:.4f}'.format(sum_loss), '{:.4f}'.format(val_loss)))
        
    return history


"""
This method computes the accuracy of the predictions of a model on the handout task, where
the two dimensional output is interpreted as a probability vector and the maximum entry is
considered as the prediction, which is then compared against the one-hot target tensors.
"""
def compute_accuracy(model, test_input, test_target, mini_batch_size=100, pytorch=False):
    
    # Error accumulator
    nb_errors = 0
    
    # Iterate over minibatches
    for b in range(0, test_input.size(0), mini_batch_size):    
        
        # Get minibatches
        test_batch = test_input.narrow(0, b, mini_batch_size)
        target_batch = test_target.narrow(0, b, mini_batch_size)

        # Forward pass
        if not pytorch:
            test_pred = model.forward(test_batch)
        else:
            test_pred = model(test_batch)
        
        # Confront targets and predictions
        max, idxmax = test_pred.max(1)
        for i in range(idxmax.shape[0]):
            if target_batch[i, idxmax[i]] != 1:
                nb_errors = nb_errors + 1
                    
    return 1 - (nb_errors/test_input.shape[0])