import numpy as np

"""
    Class that is able to generate time series.

    Uses static methods and is stateless for simplicity.
"""


class DataGenerator:

    @staticmethod
    def generate_sine(N=10000, start=0, end=np.pi):
        """
            Generates the classical sine function time series.
        """
        x = np.linspace(start, end, N)
        return np.sin(x)

    @staticmethod
    def generate_mg_euler(N, start_value, beta, gamma, n, tau):
        """
            Uses euler discretisation to generate solution to the mackey
            glass equations.
        """
        x = np.zeros(N + tau) + start_value
        for t in range(tau, N + tau):
            x[t] = x[t - 1] + (beta * x[t - tau]) / \
                (1 + np.power(x[t - tau], n)) - gamma * x[t - 1]

        return x

    @staticmethod
    def generate_macke_glass_training_data(data):
        """
        Creates the mackey glass data as specified in carabello16.

        The offset relates to the data that should be learned (the target), i.e. how
        far the prediction should look into the future.

        :param data:
        :param offset:
        :return:
        """
        x = []
        y = []
        offset = 6
        for i in range(18, data.shape[0] - offset):
            x.append(
                np.array([data[i], data[i - 6], data[i - 12], data[i - 18]]))
            y.append(np.array([data[i + offset]]))

        return np.array(x), np.array(y)

    @staticmethod
    def generate_macke_glass_euler_noisy(steps, sigma, start_value, beta, gamma, n, tau):
        """
        Generates a solution for the mackey glass time series.
        Incorporates accumulating noise model, i.e. noise in one timestep influences the next timestep.

        :param start_value:
        :param steps:
        :param beta:
        :param gamma:
        :param n:
        :param tau:
        :return:
        """
        x = np.zeros(steps + tau) + start_value
        for t in range(tau, steps + tau):
            x[t] = x[t - 1] + (beta * x[t - tau]) / \
                (1 + np.power(x[t - tau], n)) - gamma * x[t - 1]
            # add the noise
            x[t] += np.random.randn(1) * sigma

        return x
