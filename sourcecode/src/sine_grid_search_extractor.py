import matplotlib.pyplot as plt
import pprint
import sys
from keras.layers import Dense, LSTM, Conv1D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import utils


class GridSearchEntry:
    def __init__(self, input_len, n_hidden, network, learning_rate, epochs,
                 sigma, noise_type, mse_error):
        self.input_len = input_len
        self.n_hidden = n_hidden
        self.network = network
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sigma = sigma
        self.noise_type = noise_type
        self.mse_error = mse_error

    def __repr__(self):
        self_dict = {
            "input_len": self.input_len,
            "n_hidden": self.n_hidden,
            "network": self.network,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "sigma": self.sigma,
            "noise_type": self.noise_type,
            "mse_error": self.mse_error
        }
        pprint.pprint(self_dict)
        return ""


def avg_error(entries, network_type=None, learning_rate=None, epochs=None, sigma=None):
    relevant_entries = entries
    if network_type is not None:
        relevant_entries = [x for x in relevant_entries
                            if x.network == network_type]
    if learning_rate is not None:
        relevant_entries = [
            x for x in relevant_entries if x.learning_rate == learning_rate]
    if epochs is not None:
        relevant_entries = [x for x in relevant_entries if x.epochs == epochs]
    if sigma is not None:
        relevant_entries = [x for x in relevant_entries if x.sigma == sigma]
    relevant_entries = [x.mse_error for x in relevant_entries]
    return sum(relevant_entries) / len(relevant_entries)


def get_best_entry(entries, sigma):
    curr_entries = [x for x in entries if x.sigma == sigma]
    curr_entries.sort(key=lambda x: x.mse_error)
    return curr_entries[0]


def perform_estimation_plot(in_seq_len, n_hidden, net_fun, learning_rate, epochs,
                            sigma, noise_type, title):
    """
        Performs one estimation of the grid search and plots the result.
    """
    global data, N_data, out_seq_len, out_file
    figure_path = os.path.join("..", "report", "figures")

    net = net_fun(in_seq_len, n_hidden)
    optim = Adam(lr=learning_rate)
    net.compile(optim, "mse")

    curr_data = np.copy(data)
    if sigma > 0:
        if noise_type == "iid":
            curr_data += np.random.randn(len(curr_data)) * sigma
        if noise_type == "wiener":
            from sine_grid_search import wiener_noise
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
    plt.clf()
    plt.plot(test, color="blue")
    plt.plot(10 + np.arange(len(y_pred)), y_pred, color="red")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("x(t)")
    plt.legend(("Ground truth", "Prediction"))
    plt.savefig(os.path.join(figure_path, title.replace(" ", "_")+".pdf"))


if __name__ == "__main__":
    N = 3000
    N_data = 1000  # each of the data sets has exactly 1000 points
    out_seq_len = 1
    file_name = sys.argv[1]

    lines = open(file_name).readlines()

    entries = []
    for line in lines[1:]:
        arr = line.split(",")
        entries.append(GridSearchEntry(input_len=int(arr[0]),
                                       n_hidden=int(arr[1]), network=arr[2], learning_rate=float(arr[3]),
                                       epochs=int(arr[4]), sigma=float(arr[5]), noise_type=arr[6],
                                       mse_error=float(arr[7])))

    # 1: extract s.t. input_len has to be 10
    entries = [x for x in entries if x.input_len == 10]

    # 2: compare architectures for noise free runs
    noise_free_entries = [x for x in entries if x.sigma < 0.00001]
    noise_free_entries.sort(key=lambda x: x.mse_error, reverse=False)
    print("MLP error: {}".format(avg_error(noise_free_entries, "mlp")))
    print("LSTM error: {}".format(avg_error(noise_free_entries, "lstm")))
    print("CNN error: {}".format(avg_error(noise_free_entries, "cnn")))

    well_trained_cnns = [x for x in entries if x.sigma < 0.00001 and x.network == "cnn"
                         and x.learning_rate == 0.1]
    well_trained_cnns.sort(key=lambda x: x.mse_error, reverse=False)
    print("CNN error (alpha=0.1): {}".format(
        avg_error(noise_free_entries, "cnn", 0.1)))
    print("CNN error (alpha=0.01): {}".format(
        avg_error(noise_free_entries, "cnn", 0.01)))
    print("CNN error (alpha=0.001): {}".format(
        avg_error(noise_free_entries, "cnn", 0.001, 100)))

    print("LSTM error (alpha=0.1): {}".format(
        avg_error(noise_free_entries, "lstm", 0.1)))
    print("LSTM error (alpha=0.01): {}".format(
        avg_error(noise_free_entries, "lstm", 0.01)))
    print("LSTM error (alpha=0.001): {}".format(
        avg_error(noise_free_entries, "lstm", 0.001, 100)))

    # 3: compare architecture for different types of noise
    noisy_entries = [x for x in entries if x.sigma > 0.00001]
    noisy_entries.sort(key=lambda x: x.mse_error, reverse=False)

    wiener_entries = [x for x in noisy_entries if x.noise_type == "wiener"]
    iid_entries = [x for x in noisy_entries if x.noise_type == "iid"]
    print("Wiener avg. error: {}".format(avg_error(wiener_entries)))
    print("Wiener avg. error (low noise): {}".format(
        avg_error(wiener_entries, sigma=0.01)))
    print("Wiener avg. error (high noise): {}".format(
        avg_error(wiener_entries, sigma=0.1)))
    print("IID avg. error: {}".format(avg_error(iid_entries)))
    print("IID avg. error (low noise): {}".format(
        avg_error(iid_entries, sigma=0.01)))
    print("IID avg. error (high noise): {}".format(
        avg_error(iid_entries, sigma=0.1)))
    wiener_entries.sort(key=lambda x: x.mse_error, reverse=True)
    print(wiener_entries[0], wiener_entries[-1])

    print("Wiener avg. error (high noise, mlp): {}".format(
        avg_error(wiener_entries, sigma=0.1, network_type="mlp")))
    print("Wiener avg. error (high noise, lstm): {}".format(
        avg_error(wiener_entries, sigma=0.1, network_type="lstm")))
    print("Wiener avg. error (high noise, cnn): {}".format(
        avg_error(wiener_entries, sigma=0.1, network_type="cnn")))
    print("Wiener avg. error (high noise, cnn, 0.001): {}".format(
        avg_error(wiener_entries, sigma=0.1, network_type="cnn",
                  learning_rate=0.001)))
    corrected_wiener_entries = [x for x in wiener_entries if not (x.learning_rate == 0.1 and
                                                                  x.network == "cnn")]
    print("Corrected wiener error (high noise): {}".format(
        avg_error(corrected_wiener_entries, sigma=0.1)))

    # 4 get best performing network for different use cases to generate
    # (3x2) large plot for full page plot
    mse_vals = [x.mse_error for x in entries]

    from sine_grid_search import mlp, cnn, lstm
    net_fun_dict = {
        "mlp": mlp,
        "lstm": lstm,
        "cnn": cnn
    }
    data = np.sin(np.linspace(0, 6*np.pi, N))
    best_noise_free = get_best_entry(noise_free_entries, sigma=0.0)
    print("Best architecture. 0.0", best_noise_free)

    best_iid_low = get_best_entry(iid_entries, sigma=0.01)
    print("Best architecture. iid. 0.01")
    print(best_iid_low)
    best_iid_high = get_best_entry(iid_entries, sigma=0.1)
    print("Best architecture. iid. 0.1")
    print(best_iid_high)

    best_wiener_low = get_best_entry(wiener_entries, sigma=0.01)
    print("Best architecture. wiener. 0.01")
    print(best_wiener_low)
    best_wiener_high = get_best_entry(wiener_entries, sigma=0.1)
    print("Best architecture. wiener. 0.1")
    print(best_wiener_high)

    titles = ["Noise free", "iid low noise", "iid high noise",
              "wiener low noise", "wiener high noise"]
    architectures = [best_noise_free, best_iid_low, best_iid_high, best_wiener_low,
                     best_wiener_high]
    # for i in range(len(titles)):
    #     title = titles[i]
    #     print("Perform {}".format(title))
    #     architecture = architectures[i]
    #     perform_estimation_plot(in_seq_len=architecture.input_len,
    #                             n_hidden=architecture.n_hidden, net_fun=net_fun_dict[
    #                                 architecture.network],
    #                             learning_rate=architecture.learning_rate, epochs=architecture.epochs,
    #                             sigma=architecture.sigma, noise_type=architecture.noise_type,
    #                             title=title)

    print(best_iid_low)
    figure_path = os.path.join("..", "report", "figures")
    plt.hist(mse_vals, bins=20)
    plt.title("MSE value distribution")
    plt.xlabel("MSE value")
    plt.ylabel("frequency")
    plt.savefig(os.path.join(figure_path, "histogram.pdf"))
    plt.show()
