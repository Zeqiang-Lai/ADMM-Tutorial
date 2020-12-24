import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import spdiags

INF = 9999999999


def get_difference_matrix(n):
    data = np.ones((2, n))
    data[1, :] = data[1, :] * -1
    diags = np.array([0, 1])
    return spdiags(data, diags, n, n).toarray()


def shrink(x, r):
    return np.sign(x) * np.maximum(np.abs(x) - r, 0)


def forward_diff(u):
    out = np.zeros(len(u))
    out[:-1] = np.diff(u)
    out[-1] = u[0] - u[-1]
    return out


def backward_diff(u):
    out = np.zeros(len(u))
    out[1:] = -np.diff(u)
    out[0] = u[-1] - u[0]
    return out


def admm_tv_1d(y, lmbda, rho, max_itr=20, tol=1e-4, gamma=1, logging=True):
    # initialize
    length = len(y)
    x = y
    z = np.zeros(length)
    mu = np.zeros(length)

    residual = INF

    eigFtF = abs(np.fft.fft([-1, 1], length)) ** 2
    D = forward_diff
    Dt = backward_diff

    # main loop
    if logging:
        print('ADMM --- 1D TV Denoising')
        print('itr \t ||x-xold|| \t ||z-vold|| \t ||mu-uold||')

    itr = 1
    while residual > tol and itr < max_itr:
        # store x, z, mu from previous iteration for residual calculation
        x_old = x
        z_old = z
        mu_old = mu

        # inversion step
        rhs = y + rho * Dt(z - mu / rho)
        lhs = 1 + rho * eigFtF
        x = np.fft.ifft(np.fft.fft(rhs) / lhs).real

        # denoising step
        z = shrink(D(x) + mu / rho, lmbda / rho)

        # update langrangian multiplier
        mu = mu + D(x) - z

        # update rho
        rho = rho * gamma

        residualx = np.sqrt(np.sum((x - x_old) ** 2)) / length
        residualz = np.sqrt(np.sum((z - z_old) ** 2)) / length
        residualmu = np.sqrt(np.sum((mu - mu_old) ** 2)) / length

        residual = residualx + residualz + residualmu

        itr = itr + 1

        if logging:
            print('%3g \t %3.5e \t %3.5e \t %3.5e' % (itr, residualx, residualz, residualmu))

    return x


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

    out = admm_tv_1d(b, lmbda=5, rho=1, max_itr=1000)

    plt.subplot(3, 1, 3)
    plt.plot(out)
    plt.show()
