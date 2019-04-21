from data_generator import DataGenerator
from models import TwoLayerNet
from utils import generate_sequences, train_val_split

"""
    Implementation of time series prediction of the continuous differentiable
    function f(x) = sin(x).
"""

if __name__ == "__main__":
    in_seq_len = 10
    out_seq_len = 1
    data_points = DataGenerator.generate_sine(N=10000)

    train_data, val_data = train_val_split(data_points, val_fraction=0.2)
    train_x, train_y = generate_sequences(train_data, in_seq_len, out_seq_len)
    val_x, val_y = generate_sequences(train_data, in_seq_len, out_seq_len)

    net = TwoLayerNet(n_input=in_seq_len, n_hidden=10, n_output=out_seq_len)
    net.compile("adam", "mse")

    history = net.train(train_x, train_y, val_x, val_y)

    prediction = net.predict(val_x)

    print("prediction done.")
