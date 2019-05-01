import matplotlib.pyplot as plt
import numpy as np
import os

from data_generator import DataGenerator
"""
    Produce a phase plot of the mackey glass time series for different
    levels of chaos.
"""

if __name__ == "__main__":
    N = 10000
    M = 9500
    tau = 15
    figure_path = os.path.join("..", "report", "figures")

    data = DataGenerator.generate_mg_euler(N=N, start_value=1.2, beta=0.2,
                                        gamma=0.1, n=10, tau=tau)[tau:]
    data = data[M:]
    plt.clf()
    plt.xlabel("x(t)")
    plt.ylabel("x(t-τ)")
    plt.title("Phase plot")
    plt.plot(data[:-tau], data[tau:], color="red")
    tau = 17
    data = DataGenerator.generate_mg_euler(N=N, start_value=1.2, beta=0.2,
                                        gamma=0.1, n=10, tau=tau)[tau:]
    data = data[M:]
    plt.plot(data[:-tau], data[tau:], color="blue")
    plt.savefig(os.path.join(figure_path, "{}.pdf".format(tau)))

    plt.clf()
    plt.xlabel("x(t)")
    plt.ylabel("x(t-τ)")
    plt.title("Phase plot")
    tau = 25
    data = DataGenerator.generate_mg_euler(N=N, start_value=1.2, beta=0.2,
                                        gamma=0.1, n=10, tau=tau)[tau:]
    data = data[M:]
    plt.plot(data[:-tau], data[tau:], color="blue")
    plt.savefig(os.path.join(figure_path, "{}.pdf".format(tau)))

