import matplotlib.pyplot as plt
import numpy as np
import os

from data_generator import DataGenerator
"""
    Produce a phase plot of the mackey glass time series for different
    levels of chaos.
"""


def get_mg_data(N, M, tau):
    data = DataGenerator \
        .generate_mg_euler(N=N, start_value=1.2,
                           beta=0.2,
                           gamma=0.1, n=10, tau=tau)[tau:]

    return data[:M]


def plot_phaseplot(tau_nonchaotic, tau_chaotic):
    data_nonchaotic = get_mg_data(N, M, tau=tau_nonchaotic)
    plt.clf()
    plt.xlabel("x(t)")
    plt.ylabel("x(t-τ)")
    plt.title("Phase plot")
    plt.plot(data_nonchaotic[:-tau_nonchaotic],
             data_nonchaotic[tau_nonchaotic:], color="red")

    data = DataGenerator.generate_mg_euler(N=N, start_value=1.2, beta=0.2,
                                           gamma=0.1, n=10, tau=tau_chaotic)[tau_chaotic:]
    data = data[M:]
    plt.plot(data[:-tau_chaotic], data[tau_chaotic:], color="blue")
    plt.legend(("τ=15", "τ=17"))
    plt.savefig(os.path.join(figure_path, "{}.pdf".format(tau_chaotic)))
    plt.show()


if __name__ == "__main__":
    N = 10000
    M = 9800
    figure_path = os.path.join("..", "report", "figures")
    tau_nonchaotic = 15
    plot_phaseplot(tau_nonchaotic=tau_nonchaotic, tau_chaotic = 17)
    plot_phaseplot(tau_nonchaotic=tau_nonchaotic, tau_chaotic = 25)
    print("Phase plots generated.")
