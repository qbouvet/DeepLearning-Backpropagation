import torch
import statistics as stat

"""
This file contains various utility classes and methods that belong to the following categories:

- exception handling
- result calculation and plotting

Authors: Albergoni, Bouvet, Feo
"""


"""
This small class contains different reusable methods to check type and shapes of
pytorch tensors. It is meant to be instantiated and assigned to the base module
of the framework, so that each class that extends the module can validate tensors easily.
"""
class TensorValidator(object):

    def __init__(self):
        None

    def validate_input_type(self, obj, expectedType):
        if obj.type() != expectedType:
            raise TypeError("Expected %s, but was %s" % (expectedType, obj.type()))

    def shapes_match(self, a, b):
        if a.shape != b.shape:
            raise ValueError("Shapes should match, but they were ", a.shape, b.shape)
            
    def match(self, a, b):
        if a != b:
            raise ValueError("Values should match, but they were ", a, b)
    

"""
This small method computes the mean and std of the entries in a list
"""
def mean_std(l):
    return stat.mean(l), stat.stdev(l)
