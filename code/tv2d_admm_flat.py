import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse import spdiags

INF = 9999999999


def get_difference_matrix(n, row, col):
    f = np.zeros((n, n))
    for i in range(n):
        f[i, i] = 1
        f[i, i + 1] = -1
        if i < n - col:
            f[i, i + col] = -1
    return f


def shrink(x, r):
    return np.sign(x) * np.maximum(np.abs(x) - r, 0)


def admm_tv_2d_flat(y, lmbda, rho, max_itr=20, tol=1e-4, gamma=1, logging=True):
    # initialize
    row, col = y.shape
    y = y.flatten()
    length = len(y)
    x = y
    z = np.zeros(length)
    mu = np.zeros(length)

    residual = INF

    I = np.identity(length)

    # difference matrix
    F = get_difference_matrix(length, row, col)
    Ft = np.transpose(F)
    FtF = Ft @ F

    eigFtF = abs(np.fft.fft([1, -1], length)) ** 2

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
        rhs = y + rho * Ft @ (z - mu / rho)
        lhs = I + rho * FtF
        x = np.linalg.inv(lhs) @ (rhs)
        # x = np.fft.ifft(np.fft.fft(y + rho * Ft @ (z - mu / rho)) / (I + rho * eigFtF)).real

        # print(np.mean(np.abs(x2 - x)))

        # denoising step
        z = shrink(F @ x + mu / rho, lmbda / rho)

        # update langrangian multiplier
        mu = mu + F @ x - z

        # update rho
        rho = rho * gamma

        residualx = np.sqrt(np.sum((x - x_old) ** 2)) / length
        residualz = np.sqrt(np.sum((z - z_old) ** 2)) / length
        residualmu = np.sqrt(np.sum((mu - mu_old) ** 2)) / length

        residual = residualx + residualz + residualmu

        itr = itr + 1

        if logging:
            print('%3g \t %3.5e \t %3.5e \t %3.5e' % (itr, residualx, residualz, residualmu))

    return x.reshape(row, col)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    gt = plt.imread('House256.png')
    plt.subplot(1, 3, 1)
    plt.imshow(gt, cmap='gray')

    img = plt.imread('house.png')
    g = img * 255
    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap='gray')

    out = admm_tv_2d_flat(g, lmbda=5, rho=1, max_itr=100)

    plt.subplot(1, 3, 3)
    plt.plot(out)
    plt.show()