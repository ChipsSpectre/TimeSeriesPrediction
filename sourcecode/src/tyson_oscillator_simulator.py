from tyson_oscillator import Tyson2StateOscillator
import gillespy
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time

import sys
sys.path[:0] = ['..']

if __name__ == '__main__':
    np.random.seed(0)
    tyson_model = Tyson2StateOscillator()

    # =============================================
    # Simulate the mode and return the trajectories
    # =============================================
    # To set up the model, first create an empty model object. Then, add
    # species and parameters as was set up above.
    t1 = time.time()
    tyson_trajectories = tyson_model.run(show_labels=False, seed=0)
    t2 = time.time()
    print("finished running in {} seconds.".format(t2-t1))   
    print(tyson_trajectories[0], tyson_trajectories[0].shape)

    from keras.models import Sequential
    from keras.layers import LSTM

    n_units = 101
    net = Sequential()
    net.add(LSTM(units=n_units, return_sequences=True))
    net.add(LSTM(units=n_units, return_sequences=False))
    net.compile("adam", "mse")
    x = np.array([2.0, 20.0, 1.0, 0.005, 0.05, 0.1, 1.0])
    x = x.reshape([1, len(x), 1])
    y = tyson_trajectories[0][:,1]
    # normalize y
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    y = y.reshape([1, len(y)])
    
    net.fit(x, y, epochs=100)

    y = y[0]

    print(y)
    pred = net.predict(x)
    pred = pred[0]
    print(pred)
    plt.plot(np.arange(len(pred)), pred)
    plt.plot(np.arange(len(y)), y)
    plt.show()

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
