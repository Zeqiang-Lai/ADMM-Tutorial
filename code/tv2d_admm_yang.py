import matplotlib.pyplot as plt
import numpy as np
from tv1d_admm import admm_tv_1d

INF = 9999999999


def shrink(x, r):
    return np.sign(x) * np.maximum(np.abs(x) - r, 0)


def admm_tv_2d(y, lmbda, rho=1.0, max_itr=20, tol=1e-4, gamma=1.0, logging=False):
    # initialize
    row, col = y.shape
    N = row, col

    x = y
    z1 = y
    z2 = y
    u1 = np.zeros_like(y)
    u2 = np.zeros_like(y)

    residual = INF

    # main loop
    if logging:
        print('ADMM --- 2D TV Denoising')
        print('itr \t ||x-xold|| \t ||z-vold|| \t ||mu-uold||')

    itr = 1
    while residual > tol and itr < max_itr:
        # store x, v, u from previous iteration for psnr residual calculation
        # x_old = x
        # v_old = v
        # u_old = u

        # inversion step
        x = (y + (u1 + rho * z1) + (u2 + rho * z2)) / (1 + 2 * rho)

        # denoising step
        for i in range(row):
            t1 = -1.0 / rho * u1[i, :] + x[i, :]
            z1[i, :] = admm_tv_1d(t1, lmbda=5, rho=1, max_itr=1, logging=False)
        for i in range(col):
            t2 = -1.0 / rho * u2[:, i] + x[:, i]
            z2[:, i] = admm_tv_1d(t2, lmbda=5, rho=1, max_itr=1, logging=False)

        # update langrangian multiplier
        u1 = u1 + rho * (z1 - x)
        u2 = u2 + rho * (z2 - x)

        # update rho
        rho = rho * gamma

        # residualx = (1 / np.sqrt(N)) * (np.sqrt(np.sum(np.sum((x - x_old) ** 2))));
        # residualv = (1 / np.sqrt(N)) * (np.sqrt(np.sum(np.sum((v - v_old) ** 2))));
        # residualu = (1 / np.sqrt(N)) * (np.sqrt(np.sum(np.sum((u - u_old) ** 2))));
        #
        # residual = residualx + residualv + residualu;

        itr = itr + 1

        # if logging:
        #     print('%3g \t %3.5e \t %3.5e \t %3.5e' % (itr, residualx, residualz, residualmu))

    return x


if __name__ == '__main__':
    img = plt.imread('house.png')
    g = img * 255
    plt.subplot(1, 2, 1)
    plt.imshow(g, cmap='gray')

    out = admm_tv_2d(g, lmbda=5, rho=10, max_itr=20)

    plt.subplot(1, 2, 2)
    plt.imshow(out, cmap='gray')
    plt.show()
