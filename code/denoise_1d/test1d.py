import numpy as np
import matplotlib.pyplot as plt

from tv1d_admm import admm_tv_1d
from tv1d_admm_fft import admm_tv_1d as admm_tv_1d_fft

if __name__ == '__main__':
    n = 200
    x = np.ones(n)
    for i in range(3):
        idx = int(np.random.random_sample(1) * n)
        k = np.random.random_sample() * 10
        x[idx // 2: idx] = k * x[idx // 2: idx]

    plt.subplot(3, 1, 1)
    plt.plot(x)
    b = x + np.random.randn(n)
    plt.subplot(3, 1, 2)
    plt.plot(b)

    out = admm_tv_1d(b, lmbda=5, rho=1, max_itr=100)