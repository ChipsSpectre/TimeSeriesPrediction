import matplotlib.pyplot as plt
import numpy as np


"""
    Contains scripts to plot different plots of the report.
"""


def plot_prediction(data, train_pred, val_pred, in_seq_len, out_file=None,
                    title=None, ylabel=None):
    """
        Plots the prediction on a time series on train and validation data.

        If an output file is provided, the plot is stored there - otherwise the
        plot is displayed directly.
    """
    n_train, n_val = len(train_pred), len(val_pred)
    n_data = len(data)

    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(data)), data, color="blue")
    plt.plot(in_seq_len + np.arange(n_train), train_pred, color="green")
    plt.plot((n_data - n_val) + np.arange(n_val), val_pred, color="red")
    plt.legend(("ground truth", "prediction (train)", "prediction (val)"))
    plt.xlabel("time step")
    plt.ylabel("time series f(t)")

    if title is not None:
        plt.title(title)

    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)


def plot_trainval_loss(history, sigma):
    N = len(history.history["loss"])
    plt.plot(1 + np.arange(N), history.history["loss"])
    plt.plot(1 + np.arange(N), history.history["val_loss"])
    plt.plot(1 + np.arange(N), np.ones(N) * (sigma * sigma))
    plt.legend(("train", "val", "variance"))
    plt.show()


def plot_loss_comparison(history1, history2, out_file = None):
    epochs = len(history1.history["loss"])
    plt.close("all")
    plt.figure(figsize=(12, 6))
    arr = np.array(history1.history["loss"])
    plt.plot(np.arange(epochs), arr)
    plt.plot(np.arange(epochs), history1.history["val_loss"])
    plt.plot(np.arange(epochs), history2.history["loss"])
    plt.plot(np.arange(epochs), history2.history["val_loss"])

    plt.legend(("train (iid)", "val (iid)", "train (mem)", "val (mem)"))
    plt.title("Loss for noisy sinus time series")
    plt.xlabel("time step")
    plt.ylabel("MSE loss")
    if out_file is not None:
        plt.savefig(out_file)
    else:
        plt.show()
