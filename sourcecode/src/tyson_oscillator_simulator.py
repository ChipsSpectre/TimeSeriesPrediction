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
    n_traj = 101
    figure_path = os.path.join("..", "report", "figures")

    tyson_model = Tyson2StateOscillator()

    # =============================================
    # Simulate the mode and return the trajectories
    # =============================================
    # To set up the model, first create an empty model object. Then, add
    # species and parameters as was set up above.
    t1 = time.time()
    tyson_trajectories = tyson_model.run(show_labels=False, seed=0,
                                         number_of_trajectories=n_traj)
    t2 = time.time()
    print("finished running in {} seconds.".format(t2-t1))

    n_units = 101
    net = Sequential()
    net.add(LSTM(units=n_units, return_sequences=True))
    net.add(LSTM(units=n_units, return_sequences=False))
    net.compile("adam", "mse")
    params = np.array([2.0, 20.0, 1.0, 0.005, 0.05, 0.1, 1.0])
    x = np.zeros([n_traj-1, len(params), 1])
    for i in range(n_traj-1):
        x[i, :, 0] = params

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

    import sys
    sys.exit(0)

    # =============================================
    # plot just the first trajectory, 0, in both time and phase space:
    # =============================================
    from matplotlib import gridspec

    gs = gridspec.GridSpec(1, 2)

    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(tyson_trajectories[0][:, 0], tyson_trajectories[0][:, 1],
             label='X')
    ax0.plot(tyson_trajectories[0][:, 0], tyson_trajectories[0][:, 2],
             label='Y')
    ax0.legend()
    ax0.set_xlabel('Time')
    ax0.set_ylabel('Species Count')
    ax0.set_title('Time Series Oscillation')

    ax1 = plt.subplot(gs[0, 1])
    ax1.plot(tyson_trajectories[0][:, 1], tyson_trajectories[0][:, 2], 'k')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Phase-Space Plot')

    plt.tight_layout()
    plt.show()
