import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

from data_generator import DataGenerator
from models import LSTMPredictor

"""
    Implements time series prediction of the mackey glasst time series using
    different types of neural networks.
"""

if __name__ == "__main__":
    np.random.seed(0)

    taus = range(1, 102, 10)
    in_seq_len = 4
    out_seq_len = 1
    n_hidden = 10
    epochs = 100
    figure_path = os.path.join("..", "report", "figures")

    net = LSTMPredictor(n_input=in_seq_len, n_hidden=10, n_output=out_seq_len)
    net.compile("adam", "mse")

    res_chaos = []
    for tau in tqdm.tqdm(taus):
        data_points = DataGenerator \
            .generate_mg_euler(N=5000, start_value=1.5,
                               beta=0.2, gamma=0.1, n=10, tau=tau)[tau:]
        train, val, test = data_points[:3000], data_points[3000:4000], \
            data_points[4000:]
        train_x, train_y = DataGenerator \
            .generate_macke_glass_training_data(data=train)
        val_x, val_y = DataGenerator \
            .generate_macke_glass_training_data(data=val)

        # prepare shape for CNN predictor
        train_x = train_x.reshape(*train_x.shape, 1)
        val_x = val_x.reshape(*val_x.shape, 1)

        history = net.train(train_x, train_y, val_x, val_y, epochs, verbose=0)

        # plt.plot(np.arange(epochs), history.history["loss"])
        # plt.plot(np.arange(epochs), history.history["val_loss"])
        # plt.show()

        res_chaos.append(history.history["val_loss"][-1])
        print(res_chaos[-1])
        print(np.array(res_chaos))

    fig, ax = plt.subplots()
    ax.set_xticks(taus)
    plt.plot(np.array(taus), np.array(res_chaos))
    plt.title("Prediction capability under chaos")
    plt.xlabel("time delay")
    plt.ylabel("validation MSE after training")
    plt.savefig(os.path.join(figure_path, "mackey_glass_cnn.pdf"))
    plt.show()
    print(res_chaos)
