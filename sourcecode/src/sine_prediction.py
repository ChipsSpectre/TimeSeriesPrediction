import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from data_generator import DataGenerator
from models import TwoLayerNet, LSTMPredictor, CNNPredictor
from noise import iid_normal_noise, memory_normal_noise
from plotting import plot_prediction, plot_trainval_loss, plot_loss_comparison
from utils import generate_sequences, train_val_split

"""
    Implementation of time series prediction of the continuous differentiable
    function f(x) = sin(x).
"""


def run_twolayer(out_file, title):
    net = TwoLayerNet(n_input=in_seq_len, n_hidden=10, n_output=out_seq_len)
    net.compile("adam", "mse")

    train_data, val_data = train_val_split(data_points, val_fraction=0.5)
    train_x, train_y = generate_sequences(train_data, in_seq_len, out_seq_len)
    val_x, val_y = generate_sequences(val_data, in_seq_len, out_seq_len)

    history = net.train(train_x, train_y, val_x, val_y, epochs=epochs)

    train_prediction = net.predict(train_x)
    val_prediction = net.predict(val_x)

    # plot results
    plot_prediction(data_points, train_prediction, val_prediction, in_seq_len,
                    out_file,
                    title)
    return history


def run_lstm(out_file, title):
    net = LSTMPredictor(in_seq_len, n_hidden, out_seq_len)
    net.compile("adam", "mse")

    train_data, val_data = train_val_split(data_points, val_fraction=0.5)
    train_x, train_y = generate_sequences(train_data, in_seq_len, out_seq_len)
    val_x, val_y = generate_sequences(val_data, in_seq_len, out_seq_len)

    train_x = train_x.reshape(*train_x.shape, 1)
    val_x = val_x.reshape(*val_x.shape, 1)

    history = net.train(train_x, train_y, val_x, val_y, epochs=epochs)

    train_prediction = net.predict(train_x)
    val_prediction = net.predict(val_x)

    # plot results
    plot_prediction(data_points, train_prediction, val_prediction, in_seq_len,
                    out_file, title)
    return history


def run_cnn(out_file, title):
    net = CNNPredictor(n_hidden, out_seq_len)
    net.compile("adam", "mse")

    train_data, val_data = train_val_split(data_points, val_fraction=0.5)
    train_x, train_y = generate_sequences(train_data, in_seq_len, out_seq_len)
    val_x, val_y = generate_sequences(val_data, in_seq_len, out_seq_len)

    train_x = train_x.reshape(*train_x.shape, 1)
    val_x = val_x.reshape(*val_x.shape, 1)

    history = net.train(train_x, train_y, val_x, val_y, epochs=epochs)

    train_prediction = net.predict(train_x)
    val_prediction = net.predict(val_x)

    # plot results
    plot_prediction(data_points, train_prediction, val_prediction, in_seq_len,
                    out_file, title)
    return history


if __name__ == "__main__":
    np.random.seed(1)

    in_seq_len = 10
    out_seq_len = 1
    sigma = 0.1
    epochs = 100
    n_hidden = 10
    N = 1000
    figure_path = os.path.join("..", "report", "figures")

    original_data = DataGenerator.generate_sine(N, start=0, end=4 * np.pi)
    original_iidnoise = iid_normal_noise(original_data) * sigma
    original_memnoise = memory_normal_noise(original_data, sigma)

    # twolayer predictions
    data_points = np.copy(original_data)
    run_twolayer(out_file=os.path.join(figure_path, "plot_twolayer_noiseless.pdf"),
                 title="Sinus prediction without noise")

    data_points = np.copy(original_data) + original_iidnoise
    history_iid = run_twolayer(out_file=os.path.join(figure_path,
                                                     "plot_twolayer_iidnoise.pdf"),
                               title="Sinus prediction with i.i.d. noise")

    data_points = np.copy(original_data) + original_memnoise
    history_mem = run_twolayer(out_file=os.path.join(figure_path,
                                                     "plot_twolayer_memnoise.pdf"),
                               title="Sinus prediction with memorizing noise")

    plot_loss_comparison(history_iid, history_mem,
                         out_file=os.path.join(figure_path,
                                               "plot_twolayer_losscompare.pdf"))

    # lstm predictions
    data_points = np.copy(original_data)
    run_lstm(out_file=os.path.join(figure_path, "plot_lstm_noiseless.pdf"),
             title="Sinus prediction without noise")

    data_points = np.copy(original_data) + original_iidnoise
    history_iid = run_lstm(out_file=os.path.join(figure_path,
                                                 "plot_lstm_iidnoise.pdf"),
                           title="Sinus prediction with i.i.d. noise")

    data_points = np.copy(original_data) + original_memnoise
    history_mem = run_lstm(out_file=os.path.join(figure_path, "plot_lstm_memnoise.pdf"),
                           title="Sinus prediction with memorizing noise")

    plot_loss_comparison(history_iid, history_mem,
                         out_file=os.path.join(figure_path,
                                               "plot_lstm_losscompare.pdf"))

    # cnn predictions
    data_points = np.copy(original_data)
    run_cnn(out_file=os.path.join(figure_path, "plot_cnn_noiseless.pdf"),
            title="Sinus prediction without noise")

    data_points = np.copy(original_data) + original_iidnoise
    history_iid = run_cnn(out_file=os.path.join(figure_path, "plot_cnn_iidnoise.pdf"),
                          title="Sinus prediction with i.i.d. noise")

    data_points = np.copy(original_data) + original_memnoise
    history_mem = run_cnn(out_file=os.path.join(figure_path, "plot_cnn_memnoise.pdf"),
                          title="Sinus prediction with memorizing noise")

    plot_loss_comparison(history_iid, history_mem,
                         out_file=os.path.join(figure_path,
                                               "plot_cnn_losscompare.pdf"))
