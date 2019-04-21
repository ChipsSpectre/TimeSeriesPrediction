
from keras.layers import Dense
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
        self.net.add(Dense(n_hidden, input_shape = (n_input,), activation="relu"))
        self.net.add(Dense(n_output, activation = None))
        
    def predict(self, x):
        return self.net.predict(x)

    def train(self, x_train, y_train, x_val, y_val):
        return self.net.fit(x_train, y_train, validation_data=(x_val, y_val),
            batch_size = 1)

    def compile(self, optimizer, loss):
        return self.net.compile(optimizer, loss)