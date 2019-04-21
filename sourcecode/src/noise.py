import numpy as np

"""
    Contains functions that generate different kinds of noise.
"""


def iid_normal_noise(data):
    """
        Adds normal distributed noise to a data signal.
        All cells of the data array are perturbed independent of each other.

        Zero mean and unit standard deviation is used.
    """
    shape = data.shape

    return data + np.random.standard_normal(size=shape)
