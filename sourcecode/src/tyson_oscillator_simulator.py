from tyson_oscillator import Tyson2StateOscillator
import gillespy
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import LSTM

import sys
import os
sys.path[:0] = ['..']

if __name__ == '__main__':
    np.random.seed(0)
    NUM_PARAMS = 7 
    n_traj = 101
    n_units = 100
    figure_path = os.path.join("..", "report", "figures")

    params = [
        {
            "P": 2.0,
            "kt": 20.0,
            "kd": 1.0,
            "a0": 0.005,
            "a1": 0.05,
            "a2": 0.1,
            "kdx": 1.0
        }
    ]

    tyson_model = Tyson2StateOscillator(timespan=n_units, parameter_values = params[0])

    t1 = time.time()
    tyson_trajectories = tyson_model.run(show_labels=False, seed=0,
                                         number_of_trajectories=n_traj)
    t2 = time.time()
    print("finished running in {} seconds.".format(t2-t1))
    print(tyson_trajectories[0])

    net = Sequential()
    net.add(LSTM(units=n_units, return_sequences=True))
    net.add(LSTM(units=n_units, return_sequences=False))
    net.compile("adam", "mse")
    x = np.zeros([n_traj-1, NUM_PARAMS, 1])
    for i in range(n_traj-1):
        ps = params[0]
        x[i, :, 0] = np.array([ps["P"], ps["kt"], ps["kd"], ps["a0"],
            ps["a1"], ps["a2"], ps["kdx"]])

    y = np.zeros([n_traj-1, n_units])
    for i in range(n_traj-1):
        y[i, :] = tyson_trajectories[i][:, 1]
    # normalize y
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    net.fit(x, y, epochs=100)

    t1 = time.time()
    pred = net.predict(x)
    t2 = time.time()
    print("LSTM prediction done in {} seconds".format(t2-t1))
    pred = pred[0]

    y_test = tyson_trajectories[n_traj-1][:, 1]
    # normalize y
    y_test = (y_test - np.min(y_test)) / (np.max(y_test) - np.min(y_test))

    y_mean = np.mean(y, axis=0)

    plt.plot(np.arange(len(y_test)), y_test)
    plt.plot(np.arange(len(pred)), pred)
    plt.plot(np.arange(len(y_mean)), y_mean)
    plt.title("Stochastic time series generation")
    plt.xlabel("time step")
    plt.ylabel("time series value X(t)")
    plt.legend(("ground truth", "prediction", "mean"))
    plt.savefig(os.path.join(figure_path, "nn_limitation.pdf"))
