import torch
import math


"""
This file contains the methods necessary for the generation of the disc dataset

Authors: Albergoni, Bouvet, Feo
"""


"""
Allows to check whether a point is inside a circle
"""
def is_inside_disc(radius, x, y, center):  
    return (math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius)


"""
Generates nb points in a given range, and assign a label based on whether they
are inside or outside the provided circle.
The labels are created as one-hot vectors, with index 0 representing "outside"
and index 1 "inside".
"""
def generate_disc_set(nb, points_range, radius, center):
    
    # Generate points
    samples = torch.FloatTensor(nb, 2).uniform_(points_range[0], points_range[1])
    
    # Assign labels
    target = torch.zeros(nb, 2)
    for i in range(nb):
        if is_inside_disc(radius, samples[i, 0], samples[i, 1], center):
            target[i, 1] = 1
        else:
            target[i, 0] = 1

    return samples, target


"""
Additional dataset-handling method that allows to standardize the input.
Currently not used in the submission code.
"""
def standardize(train_input, val_input, test_input, mean, std):
    train_input.sub_(mean).div_(std)
    val_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)
    
    return train_input, val_input, test_input