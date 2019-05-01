from data_generator import DataGenerator
import matplotlib.pyplot as plt
import numpy as np


def psr_deneme(x, m, tao, npoint=None):
    """
        Phase space reconstruction.

        :param x: the time series
        :param m: the embedding dimension
        :param tao: the time delay
    """
    N = len(x)
    M = N - (m-1) * tao

    Y = np.zeros([M, m])
    for i in range(M):
        for j in range(m):
            Y[i, j] = x[i + j * tao]
    return Y


def lyarosenstein(x, m, tao, meanperiod, maxiter):
    """
        Implementation of rosenstein algorithm.

        :param x: the 1D time series
        :param m: the minimum embedding dimension
        :param tao: the time lag
        :param meanperiod: mean frequency of the power spectrum
    """
    N = len(x)
    M = N - (m-1) * tao
    Y = psr_deneme(x, m, tao)

    nearpos = np.zeros(M, dtype=np.int)

    for i in range(M):
        x0 = np.ones([M, 1]) @ Y[i, :].reshape([1, tao])
        distance = np.sqrt(np.sum(np.square(Y - x0), axis=1))
        for j in range(M):
            if abs(j-i) <= meanperiod:
                distance[j] = 1e10
        nearpos[i] = np.argmin(distance)

    d = np.zeros(maxiter)
    for k in range(maxiter):
        maxind = M - k
        evolve = 0
        pnt = 0
        for j in range(M):
            if j < maxind and nearpos[j] < maxind:
                diff = Y[j+k, :] - Y[nearpos[j] + k, :]
                dist_k = np.sqrt(np.sum(np.square(diff)))
                if dist_k != 0:
                    evolve = evolve + np.log(dist_k)
                    pnt = pnt + 1
        if pnt > 0:
            d[k] = evolve / pnt

    # plt.plot(d)
    # plt.show()
    # LLE calculation
    F = np.polyfit(range(maxiter), d, deg=1)
    lle = F[0]
    print(F, lle)
    return lle


if __name__ == "__main__":
    for tau in [12, 17, 22, 27]:
        x = DataGenerator.generate_mg_euler(N=1000, start_value=1.2,
                                            beta=0.2, gamma=0.1, n=10, tau=tau)
        # plt.plot(x)
        # plt.show()
        tao = 2
        m = 2

        Y = psr_deneme(x, tao, m)

        lyarosenstein(x, m, tao, meanperiod=2, maxiter=100)
