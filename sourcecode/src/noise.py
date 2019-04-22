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


def memory_normal_noise(data, sigma = 1):
    """
        Adds noise where the last state of the noise depends on the previous
        noise value only.

        Zero mean and customizable standard deviation is used in each time step.


        Only 1-dimensional signals are supported.
    """
    N = len(data)
    values = []
    prev = np.random.standard_normal(1) * sigma
    for _ in range(N):
        values.append(prev)
        prev += np.random.standard_normal(1) * sigma
    return np.array(values)[:, 0]
