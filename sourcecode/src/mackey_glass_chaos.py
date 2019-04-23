
import matplotlib.pyplot as plt
import numpy as np
import os

from data_generator import DataGenerator
"""
    Script that simulates the mackey glass time series equation for certain 
    conditions and shows the dependence on the initial conditions 
    by comparing 2 solutions with similar but not equal initial conditions.
"""
def plt_mg_chaos(tau, out_file):
    N = 1000
    start_one = 1.5
    data_one = DataGenerator.generate_mg_euler(N = N, start_value = start_one, beta = 0.2, 
        gamma = 0.1, n = 10, tau = tau)[tau:]
    start_two = 1.51
    data_two = DataGenerator.generate_mg_euler(N = N, start_value = start_two, beta = 0.2, 
        gamma = 0.1, n = 10, tau = tau)[tau:]
    plt.close("all")
    plt.plot(np.arange(len(data_one)), data_one)
    plt.plot(np.arange(len(data_two)), data_two)
    plt.ylabel("x(t)")
    plt.xlabel("time steps")
    plt.title("Chaos in Mackey Glass equations")
    plt.savefig(out_file)


if __name__ == "__main__":

    figure_path = os.path.join("..", "report", "figures")

    plt_mg_chaos(tau = 17, out_file = os.path.join(figure_path, "mg_chaos_17.pdf"))
    plt_mg_chaos(tau = 25, out_file = os.path.join(figure_path, "mg_chaos_25.pdf"))

