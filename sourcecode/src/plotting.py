import matplotlib.pyplot as plt
import numpy as np


"""
    Contains scripts to plot different plots of the report.
"""

def plot_prediction(data, train_pred, val_pred):
    n_train, n_val = len(train_pred), len(val_pred)
    n_data = len(data)

    plt.plot(np.arange(len(data)), data, color="blue")
    plt.plot(np.arange(n_train), train_pred, color="green")
    plt.plot((n_data - n_val) + np.arange(n_val), val_pred, color="red")
    plt.legend(("ground truth", "prediction (train)", "prediction (val)"))

    plt.show()