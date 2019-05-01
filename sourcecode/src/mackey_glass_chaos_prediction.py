
import matplotlib.pyplot as plt
import numpy as np

from data_generator import DataGenerator
from mackey_glass_prediction2 import MGCNN, np_rmse, rmse

def estimate_chaos(tau):
    N = 1500
    x_0 = 1.2
    beta = 0.2
    gamma = 0.1
    n = 10
    epochs = 100

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


    # prepare for LSTM/CNN
    train_x = train_x.reshape(*train_x.shape, 1)
    val_x = val_x.reshape(*val_x.shape, 1)
    test_x = test_x.reshape(*test_x.shape, 1)

    net = MGCNN()
    net.compile("adam", rmse)
    net.train(train_x, train_y, val_x, val_y, epochs=epochs, verbose=0)
    pred_test = net.predict(test_x)

    real_x = test_x
    preds = []
    for i in range(10):
        curr_x = real_x + i * 0.001
        pred = net.predict(curr_x)
        preds.append(np.sum(np.square(pred)))
    preds = np.array(preds)
    print("Tau=", tau, "mean=", np.mean(preds), "std=", np.std(preds))
    return np.std(preds)

if __name__ == "__main__":
    n_stat = 10
    for tau in [15, 17, 19, 21]:
        stds = []
        for _ in range(n_stat):
            stds.append(estimate_chaos(tau=tau))
        stds = np.array(stds)
        print("Mean standard deviation with tau = {}: {}".format(tau, np.mean(stds)))