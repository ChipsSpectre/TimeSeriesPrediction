import numpy as np
import os

from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, Flatten
from keras import optimizers

from data_generator import DataGenerator
from keras import backend as K

import matplotlib.pyplot as plt

"""
    Implements a 6-step in the future forecasting of the mackey glass 
    time series.

    An LSTM is used and optimized for this task.
"""
class MGMLP(Model):
    def __init__(self):
        super(MGMLP, self).__init__()
        self.name = "MLP"
        self.net = Sequential()
        self.net.add(Dense(32, activation="relu"))
        self.net.add(Dense(16, activation="relu"))
        self.net.add(Dense(1, activation=None))

    def predict(self, x):
        return self.net.predict(x)

    def train(self, x_train, y_train, x_val, y_val, epochs, verbose=1):
        return self.net.fit(x_train, y_train, validation_data=(x_val, y_val),
                            batch_size=32, epochs=epochs, verbose=verbose)

    def compile(self, optimizer, loss):
        return self.net.compile(optimizer, loss)



class MGLSTM(Model):
    """
        Implementation of LSTM model for time series prediction.

        Uses only a single hidden LSTM layer.
    """

    def __init__(self):
        super(MGLSTM, self).__init__()
        self.net = Sequential()
        self.name = "LSTM"
        self.net.add(LSTM(10, return_sequences=False))
        # self.net.add(Dense(32, activation="relu"))
        # # self.net.add(Dropout(0.2))
        # self.net.add(Dense(16, activation="relu"))
        self.net.add(Dense(1, activation=None))

    def predict(self, x):
        return self.net.predict(x)

    def train(self, x_train, y_train, x_val, y_val, epochs, verbose=1):
        return self.net.fit(x_train, y_train, validation_data=(x_val, y_val),
                            batch_size=32, epochs=epochs, verbose=verbose)

    def compile(self, optimizer, loss):
        return self.net.compile(optimizer, loss)


class MGCNN(Model):

    def __init__(self):
        super(MGCNN, self).__init__()
        self.net = Sequential()
        self.name = "CNN"
        n_filters = 8
        self.net.add(Conv1D(filters=n_filters,
                            activation="relu", kernel_size=3, padding="same"))
        self.net.add(Conv1D(filters=n_filters,
                            activation="relu", kernel_size=3, padding="same"))
        self.net.add(Conv1D(filters=n_filters,
                            activation="relu", kernel_size=3, padding="same"))
        self.net.add(Flatten())
        self.net.add(Dense(1, activation=None))

    def predict(self, x):
        return self.net.predict(x)

    def train(self, x_train, y_train, x_val, y_val, epochs, verbose=1):
        return self.net.fit(x_train, y_train, validation_data=(x_val, y_val),
                            batch_size=32, epochs=epochs, verbose=verbose)

    def compile(self, optimizer, loss):
        return self.net.compile(optimizer, loss)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def np_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

if __name__ == "__main__":
    np.random.seed(0)

    in_seq_len = 4
    out_seq_len = 1
    epochs = 1000
    figure_path = os.path.join("..", "report", "figures")

    net = MGLSTM()
    optim = optimizers.Adam()
    net.compile(optim, rmse)

    # use same time series parameters as Caraballo et al.
    tau = 17
    data_points = DataGenerator \
        .generate_mg_euler(N=1500, start_value=1.2,
                           beta=0.2, gamma=0.1, n=10, tau=tau)[tau:]
    train, val, test = data_points[:500], data_points[500:1000], \
        data_points[1000:]
    train_x, train_y = DataGenerator \
        .generate_macke_glass_training_data(data=train)
    val_x, val_y = DataGenerator \
        .generate_macke_glass_training_data(data=val)
    test_x, test_y = DataGenerator \
        .generate_macke_glass_training_data(data=test)

    # prepare for LSTM/CNN
    # train_x = train_x.reshape(*train_x.shape, 1)
    # val_x = val_x.reshape(*val_x.shape, 1)
    # test_x = test_x.reshape(*test_x.shape, 1)

    history = net.train(train_x, train_y, val_x, val_y, epochs, verbose=1)

    # prediction training
    pred_train = net.predict(train_x)
    pred_val = net.predict(val_x)
    pred_test = net.predict(test_x)

    print("Test error: {}".format(np_rmse(test_y, pred_test)))

    # plt.figure(figsize=(20,10))
    # plt.plot(np.arange(data_points.shape[0]), data_points)
    # plt.plot(23 + np.arange(pred_train.shape[0]), pred_train)
    # plt.plot(500 + 23 + np.arange(pred_val.shape[0]), pred_val)
    # plt.plot(1000 + 23 + np.arange(pred_test.shape[0]), pred_test)
    # plt.title("Noiseless Mackey Glass time series forecasting")
    # plt.xlabel("time step")
    # plt.ylabel("time series value x(t)")
    # plt.legend(("ground truth", "train pred.",
    #             "validation pred.", "test pred."))
    # plt.savefig(os.path.join(figure_path, "mg_pred_cnn.pdf"))
    # plt.show()
