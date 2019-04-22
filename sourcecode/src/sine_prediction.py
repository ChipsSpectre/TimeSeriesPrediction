import matplotlib.pyplot as plt
import numpy as np

from data_generator import DataGenerator
from models import TwoLayerNet, LSTMPredictor
from noise import iid_normal_noise, memory_normal_noise
from plotting import plot_prediction, plot_trainval_loss
from utils import generate_sequences, train_val_split

"""
    Implementation of time series prediction of the continuous differentiable
    function f(x) = sin(x).
"""


def run_twolayer():
    net = TwoLayerNet(n_input=in_seq_len, n_hidden=10, n_output=out_seq_len)
    net.compile("adam", "mse")

    data_points = DataGenerator.generate_sine(N=1000, start=0, end=4 * np.pi)
    # data_points = data_points + iid_normal_noise(data_points) * sigma
    data_points = data_points + memory_normal_noise(data_points, sigma)

    train_data, val_data = train_val_split(data_points, val_fraction=0.5)
    train_x, train_y = generate_sequences(train_data, in_seq_len, out_seq_len)
    val_x, val_y = generate_sequences(val_data, in_seq_len, out_seq_len)

    history = net.train(train_x, train_y, val_x, val_y, epochs=epochs)

    train_prediction = net.predict(train_x)
    val_prediction = net.predict(val_x)

    # plot results
    plot_prediction(data_points, train_prediction, val_prediction)

    # plot the loss compared to variance of the noise
    plot_trainval_loss(history, sigma)


def run_lstm():
    n_hidden = 100
    net = LSTMPredictor(in_seq_len, n_hidden, out_seq_len)
    net.compile("adam", "mse")

    data_points = DataGenerator.generate_sine(N=1000, start=0, end=4 * np.pi)
    # data_points = data_points + iid_normal_noise(data_points) * sigma
    data_points = data_points + memory_normal_noise(data_points, sigma)

    train_data, val_data = train_val_split(data_points, val_fraction=0.5)
    train_x, train_y = generate_sequences(train_data, in_seq_len, out_seq_len)
    val_x, val_y = generate_sequences(val_data, in_seq_len, out_seq_len)

    train_x = train_x.reshape(*train_x.shape, 1)
    val_x = val_x.reshape(*val_x.shape, 1)

    history = net.train(train_x, train_y, val_x, val_y, epochs=epochs)

    train_prediction = net.predict(train_x)
    val_prediction = net.predict(val_x)

    # plot results
    plot_prediction(data_points, train_prediction, val_prediction)

    # plot the loss compared to variance of the noise
    plot_trainval_loss(history, sigma)


if __name__ == "__main__":
    in_seq_len = 10
    out_seq_len = 1
    sigma = 0.1
    epochs = 100
    n_hidden = 10

    run_lstm()
    run_twolayer()