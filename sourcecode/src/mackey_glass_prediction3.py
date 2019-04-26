import numpy as np
import matplotlib.pyplot as plt
import os

from data_generator import DataGenerator
from mackey_glass_prediction2 import MGCNN, MGLSTM, MGMLP, np_rmse, rmse

if __name__ == "__main__":
    np.random.seed(0)
    tau = 17
    N = 1500
    x_0 = 1.2
    beta = 0.2
    n = 10
    gamma = 0.1

    epochs = 200
    figure_path = os.path.join("..", "report", "figures")
    out_file = "result.txt"
    f = open(out_file, "w+")

    data_points = DataGenerator.generate_mg_euler(N=N, start_value=x_0,
                                                  beta=beta, gamma=gamma, n=n, tau=tau)[tau:]
    train, val, test = data_points[:500], data_points[500:1000], \
        data_points[1000:]
    train_x, train_y = DataGenerator \
        .generate_macke_glass_training_data(data=train)
    val_x, val_y = DataGenerator \
        .generate_macke_glass_training_data(data=val)
    test_x, test_y = DataGenerator \
        .generate_macke_glass_training_data(data=test)

    nets = [MGCNN(), MGMLP(), MGLSTM()]
    sigmas = [0.01, 0.1]
    noise_types = ["iid", "wiener"]

    for net in nets:
        for noise_type in noise_types:
            for sigma in sigmas:
                # prepare data
                curr_points = np.copy(data_points)
                curr_train_x, curr_train_y = np.copy(train_x), np.copy(train_y)
                curr_val_x, curr_val_y = np.copy(val_x), np.copy(val_y)
                curr_test_x, curr_test_y = np.copy(test_x), np.copy(test_y)
                # compile net
                net.compile("adam", rmse)
                # apply noise
                if noise_type == "iid":
                    curr_train_x += np.random.standard_normal(
                        size=curr_train_x.shape) * sigma
                    curr_val_x += np.random.standard_normal(
                        size=curr_val_x.shape) * sigma
                    curr_test_x += np.random.standard_normal(
                        size=curr_test_x.shape) * sigma
                if noise_type == "wiener":
                    curr_points = DataGenerator \
                        .generate_macke_glass_euler_noisy(steps=N, sigma=sigma,
                                                          start_value=x_0, beta=beta, gamma=gamma, n=n, tau=tau)
                    train, val, test = curr_points[:500], curr_points[500:1000], \
                        curr_points[1000:]
                    curr_train_x, curr_train_y = DataGenerator \
                        .generate_macke_glass_training_data(data=train)
                    curr_val_x, curr_val_y = DataGenerator \
                        .generate_macke_glass_training_data(data=val)
                    curr_test_x, curr_test_y = DataGenerator \
                        .generate_macke_glass_training_data(data=test)
                # adapt data format to architecture
                if type(net).__name__ != "MGMLP":
                    curr_train_x = curr_train_x.reshape(*curr_train_x.shape, 1)
                    curr_val_x = curr_val_x.reshape(*curr_val_x.shape, 1)
                    curr_test_x = curr_test_x.reshape(*curr_test_x.shape, 1)

                net.train(x_train=curr_train_x, y_train=curr_train_y,
                          x_val=curr_val_x, y_val=curr_val_y, epochs=epochs)

                pred_y = net.predict(curr_test_x)
                rmse_val = np_rmse(curr_test_y, pred_y)

                result = {
                    "name" : net.name,
                    "noise" : noise_type,
                    "sigma" : sigma,
                    "rmse" : rmse_val
                }
                f.write(str(result)+"\n")
                print("Test error: {}".format(str(result)))