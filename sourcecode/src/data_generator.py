import numpy as np

"""
    Class that is able to generate time series.

    Uses static methods and is stateless for simplicity.
"""

class DataGenerator:

    @staticmethod
    def generate_sine(N = 10000, start = 0, end = np.pi):
        """
            Generates the classical sine function time series.
        """
        x = np.linspace(start, end, N)
        return np.sin(x)

