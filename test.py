# Torch imports
import torch
from torch import nn as torch_nn
from torch.autograd import Variable

# Framework imports
import nn, losses, optim
import activations as F

# Utility imports
import math
import time
from utils.helpers import mean_std
from utils.dataset_generation import generate_disc_set, standardize
from utils.training import train_model, compute_accuracy


"""
This file presents an example of the small deep learning framework implemented in the context of this project.
It does so by training the requested net (see handout) on the disc toy dataset. It also trains an equivalent
pure pytorch model, which serves as a reference to evaluate our framework.

Both models are trained 10 times each, with new data and re-initialized parameters each time, and the final
results are then presented.

Authors: Albergoni, Bouvet, Feo
"""


"""
This method initializes the net requested by the project handout with framework's modules and
trains it on the provided dataset.
"""
def train_framework(train_set, val_set, test_set, epochs=1000, mini_batch_size=50, lr=0.001):

    # Turn autograd off
    torch.set_grad_enabled(False)
    
    # Net definition
    model = nn.Sequential(
        nn.Linear(2, 25, activation="relu"),
        F.ReLU(),
        nn.Linear(25, 25, activation="relu"),
        F.ReLU(),
        nn.Linear(25, 25, activation="relu"),
        F.ReLU(),
        nn.Linear(25, 2, activation="relu")
    )
    
    # Training params
    opt = optim.SGD(lr, model)
    criterion = losses.LossMSE()
    
    # Train
    start_time = time.perf_counter()
    history = train_model(model,
                          train_set[0], train_set[1], val_set[0], val_set[1],
                          criterion, opt, epochs, mini_batch_size,
                          pytorch=False, verbose=True)
    end_time = time.perf_counter()
    
    # Compute final accuracies
    train_acc = compute_accuracy(model, train_set[0], train_set[1], pytorch=False)
    test_acc = compute_accuracy(model, test_set[0], test_set[1], pytorch=False)
    print("\tTraining time : %s s" % (end_time - start_time))
    print("\tAccuracy : train_acc = %s \t test_acc = %s" % (train_acc, test_acc))
    
    return history, end_time - start_time, (train_acc, test_acc)
    

"""
This method initializes the net requested by the project handout with pytorch's modules and
trains it on the provided dataset.
"""
def train_pytorch(train_set, val_set, test_set, epochs=1000, mini_batch_size=50, lr=0.001):

    # Turn autograd on
    torch.set_grad_enabled(True)

    # Net definition
    model = torch_nn.Sequential(
        torch_nn.Linear(2, 25),
        torch_nn.ReLU(),
        torch_nn.Linear(25, 25),
        torch_nn.ReLU(),
        torch_nn.Linear(25, 25),
        torch_nn.ReLU(),
        torch_nn.Linear(25, 2)
    )
        
    # Training params
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch_nn.MSELoss(reduction='sum')
    
    # Train and present results
    start_time = time.perf_counter()
    history = train_model(model,
                          Variable(train_set[0]), Variable(train_set[1]),
                          Variable(val_set[0]), Variable(val_set[1]),
                          criterion, opt, epochs, mini_batch_size,
                          pytorch=True, verbose=True)
    end_time = time.perf_counter()
    
    # Compute final accuracies
    train_acc = compute_accuracy(model, Variable(train_set[0]), Variable(train_set[1]), pytorch=True)
    test_acc = compute_accuracy(model, Variable(test_set[0]), Variable(test_set[1]), pytorch=True)
    print("\tTraining time : %s s" % (end_time - start_time))
    print("\tAccuracy : train_acc = %s \t test_acc = %s" % (train_acc, test_acc))
    
    return history, end_time - start_time, (train_acc, test_acc)


"""
Main method
"""
def main():
    
    # Disc parameters
    center = (0.5, 0.5)
    points_range = (0, 1)
    radius = 1 / math.sqrt(2 * math.pi)
    
    # Train 10 times the two models
    num_training_runs = 10
    framework_histories = []
    framework_times = []
    framework_accuracies = []
    pytorch_histories = []
    pytorch_times = []
    pytorch_accuracies = []
    
    for i in range(num_training_runs):
        
        # Generation of the disc dataset according to the handout request
        train_set = generate_disc_set(1000, points_range, radius, center)  
        val_set = generate_disc_set(1000, points_range, radius, center)
        test_set = generate_disc_set(1000, points_range, radius, center)

        print("> Training run %s/%s\n" % (i+1, num_training_runs))
        
        # Train framework model
        print("\tFramework model")
        framework_res = train_framework(train_set, val_set, test_set, lr=0.001, epochs=1000)
        framework_histories.append(framework_res[0])
        framework_times.append(framework_res[1])
        framework_accuracies.append(framework_res[2])
        print()
        
        # Train pytorch model
        print("\tPytorch model")
        pytorch_res = train_pytorch(train_set, val_set, test_set, lr=0.001, epochs=1000)
        pytorch_histories.append(pytorch_res[0])
        pytorch_times.append(pytorch_res[1])
        pytorch_accuracies.append(pytorch_res[2])
        print()
        
    # Compute results  
    fr_mean, fr_std = mean_std([accs[1] for accs in framework_accuracies])
    py_mean, py_std = mean_std([accs[1] for accs in pytorch_accuracies])
    fr_time_mean, fr_time_std = mean_std(framework_times)
    py_time_mean, py_time_std = mean_std(pytorch_times)
    print("> Final results")
    print("\tTest accuracy of framework model : \t mean = %s \t std = %s" % (fr_mean, fr_std))
    print("\tMean train time of framework model : \t %s s \t std = %s" % (fr_time_mean, fr_time_std))
    print("\tTest accuracy of pytorch model : \t mean = %s \t std = %s" % (py_mean, py_std))
    print("\tMean train time of pytorch model : \t %s s \t std = %s" % (py_time_mean, py_time_std))
    print()
    
    print("NOTE: the plots visible on the report are not generated by the code you just run,"+ \
          "but by code in a separate Jupyter Notebook that you can also find in the submission folder." + \
          "This was done in order to use seaborn (not available on the VM) for the plots.")

if __name__ == "__main__":
    main()




