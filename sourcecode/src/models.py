
from keras.layers import Dense, LSTM, Conv1D, Flatten
from keras.models import Sequential, Model

"""
    Collection of different keras models.

    All models here are intended for time series prediction.
"""


class TwoLayerNet(Model):
    """
        Implementation of simplistic time series predictor.

        Uses only one hidden layer, activation function is ReLU.
    """

    def __init__(self, n_input, n_hidden, n_output):
        super(TwoLayerNet, self).__init__()
        self.net = Sequential()
        self.net.add(Dense(n_hidden, input_shape=(
            n_input,), activation="relu"))
        self.net.add(Dense(n_output, activation=None))

    def predict(self, x):
        return self.net.predict(x)

    def train(self, x_train, y_train, x_val, y_val, epochs, verbose=1):
        return self.net.fit(x_train, y_train, validation_data=(x_val, y_val),
                            batch_size=128, epochs=epochs, verbose=verbose)

    def compile(self, optimizer, loss):
        return self.net.compile(optimizer, loss)


class LSTMPredictor(Model):
    """
        Implementation of LSTM model for time series prediction.

        Uses only a single hidden LSTM layer.
    """

    def __init__(self, n_input, n_hidden, n_output):
        super(LSTMPredictor, self).__init__()
        self.net = Sequential()
        self.net.add(LSTM(n_hidden, return_sequences=False))
        self.net.add(Dense(n_output, activation=None))

    def predict(self, x):
        return self.net.predict(x)

    def train(self, x_train, y_train, x_val, y_val, epochs, verbose=1):
        return self.net.fit(x_train, y_train, validation_data=(x_val, y_val),
                            batch_size=128, epochs=epochs, verbose=verbose)

    def compile(self, optimizer, loss):
        return self.net.compile(optimizer, loss)


class CNNPredictor(Model):
    """
        Implementation of a time series predictor based on convolutional layers
        for feature extraction. The final regression is performed by a fully
        connected layer.
    """

    def __init__(self, n_filters, n_output):
        super(CNNPredictor, self).__init__()
        self.net = Sequential()
        self.net.add(Conv1D(filters=n_filters,
                            activation="relu", kernel_size=3, padding="same"))
        self.net.add(Conv1D(filters=n_filters,
                            activation="relu", kernel_size=3, padding="same"))
        self.net.add(Flatten()),
        self.net.add(Dense(n_output, activation=None))

    def predict(self, x):
        return self.net.predict(x)

    def train(self, x_train, y_train, x_val, y_val, epochs, verbose=1):
        return self.net.fit(x_train, y_train, validation_data=(x_val, y_val),
                            batch_size=128, epochs=epochs, verbose=verbose)

    def compile(self, optimizer, loss):
        return self.net.compile(optimizer, loss)
