from keras.layers import Dense, LSTM, Conv1D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import utils
"""
    Grid search for sinus approximation.
"""


def mlp(n_input, n_hidden, n_output=1):
    """
        Creates a multi layer perceptron with one hidden layer using keras.
    """
    net = Sequential()
    net.add(Dense(n_hidden, input_shape=(n_input,), activation="relu"))
    net.add(Dense(n_output, activation=None))
    return net


def lstm(n_input, n_hidden, n_output=1):
    """
        Creates a LSTM architecture with one hidden layer.
    """
    net = Sequential()
    net.add(LSTM(n_hidden, return_sequences=False))
    net.add(Dense(n_output, activation=None))
    return net


def cnn(n_input, n_hidden, n_output=1):
    """
        Creates a convolutional neural network with the number of filters
        equal to n_hidden.
    """
    net = Sequential()
    for _ in range(n_hidden):
        net.add(Conv1D(filters=8,
                       activation="relu", kernel_size=3, padding="same"))
    net.add(Flatten()),
    net.add(Dense(n_output, activation=None))
    return net


def wiener_noise(n, sigma):
    """
        Generates wiener process, i.e. a brownian motion for n steps with a 
        certain standard deviation.
    """
    res = np.zeros(n)
    for i in range(1, n):  # first per definition always zero
        res[i] = res[i-1] + np.random.randn(1) * sigma
    return res


def perform_estimation(in_seq_len, n_hidden, net_fun, learning_rate, epochs,
                       sigma, noise_type):
    """
        Performs one estimation of the grid search.
    """
    global data, N_data, out_seq_len, out_file
    net = net_fun(in_seq_len, n_hidden)
    optim = Adam(lr=learning_rate)
    net.compile(optim, "mse")

    curr_data = np.copy(data)
    if sigma > 0:
        if noise_type == "iid":
            curr_data += np.random.randn(len(curr_data)) * sigma
        if noise_type == "wiener":
            curr_data += wiener_noise(len(curr_data), sigma)

    train, val, test = curr_data[:N_data], curr_data[N_data:2 *
                                                     N_data], curr_data[2*N_data:]
    x, y = utils.generate_sequences(
        train, in_seq_len, out_seq_len)
    x_val, y_val = utils.generate_sequences(
        val, in_seq_len, out_seq_len)
    x_test, y_test = utils.generate_sequences(
        test, in_seq_len, out_seq_len)
    if net_fun != mlp:
        # prepare data for LSTM/CNN
        x = x.reshape(*x.shape, 1)
        x_val = x_val.reshape(*x_val.shape, 1)
        x_test = x_test.reshape(*x_test.shape, 1)
    net.fit(x, y, validation_data=(x_val, y_val),
            epochs=epochs, verbose=0)
    y_pred = net.predict(x_test)
    mse = np.square(y_test - y_pred).mean()
    out_str = ("{},{},{},{},{},{},{},{}\n".format(in_seq_len, n_hidden, net_fun.__name__,
                                        learning_rate, epochs, sigma, noise_type, mse))
    open(out_file, mode="a").write(out_str)
    print(out_str)


def grid_search():
    in_seq_lens = [5]
    n_hiddens = [10]
    net_funs = [mlp, lstm, cnn]
    learning_rates = [0.001]
    epochs_list = [1]
    sigmas = [0]
    noise_types = ["iid", "wiener"]

    # original data
    for in_seq_len in in_seq_lens:
        for n_hidden in n_hiddens:
            for net_fun in net_funs:
                for learning_rate in learning_rates:
                    for epochs in epochs_list:
                        for sigma in sigmas:
                            for noise_type in noise_types:
                                perform_estimation(in_seq_len, n_hidden, net_fun,
                                                   learning_rate, epochs, sigma, noise_type)
    print("Grid search completed.")


if __name__ == "__main__":
    N = 3000
    N_data = 1000  # each of the data sets has exactly 1000 points
    out_seq_len = 1
    data = np.sin(np.linspace(0, 6*np.pi, N))
    out_file = "sine_results.txt"
    grid_search()
