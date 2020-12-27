import numpy as np


def PSNR(x, gt, data_range=1):
    mse = np.mean((x - gt) ** 2)
    return 10 * np.log10(data_range ** 2 / mse)


def forward_diff(u):
    dux = np.column_stack([u[:, 0] - u[:, -1], np.diff(u, axis=1)])
    duy = np.row_stack([u[0, :] - u[-1, :], np.diff(u, axis=0)])
    return dux, duy


def dive(x, y):
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=0)
    # dtxy = np.column_stack([x[:, -1] - x[:, 0], -dx])
    # dtxy = dtxy + np.row_stack([y[-1, :] - y[0, :], -dy])
    dtxy = np.column_stack([-dx, x[:, -1] - x[:, 0]])
    dtxy = dtxy + np.row_stack([-dy, y[-1, :] - y[0, :]])
    return dtxy


def defDDt():
    D = lambda u: forward_diff(u)
    Dt = lambda x, y: dive(x, y)
    return D, Dt


def shrink(x, r):
    return np.sign(x) * np.maximum(np.abs(x) - r, 0)


def admm(y, lam, rho, Nit, tol=1e-4, logging=True):
    row, col = y.shape
    x = y
    alpha = 0.7
    v1 = np.zeros((row, col))
    v2 = np.zeros((row, col))

    y1 = np.zeros((row, col))
    y2 = np.zeros((row, col))

    eigDtD = np.abs(np.fft.fft2(np.array([[1, -1]]), (row, col))) ** 2 + \
             np.abs(np.fft.fft2(np.array([[1, -1]]).transpose(), (row, col))) ** 2

    D, Dt = defDDt()

    # curNorm = np.linalg.norm(Dx1 - v1) + np.linalg.norm(Dx2 - v2)

    if logging:
        print('ADMM --- 2D TV Denoising')
        print('itr \t ||x-xold||')

    imgs = []

    last_iter = Nit
    for k in range(Nit):
        imgs.append(x)
        x_old = x

        ty1 = v1 - y1 / rho
        ty2 = v2 - y2 / rho
        tmp = Dt(ty1, ty2)
        rhs = y + rho * tmp
        lhs = 1 + rho * eigDtD

        x = np.fft.fft2(rhs) / lhs
        x = np.fft.ifft2(x).real

        Dx1, Dx2 = D(x)

        v1 = shrink(Dx1 + y1 / rho, lam / rho)
        v2 = shrink(Dx2 + y2 / rho, lam / rho)

        y1 = y1 + Dx1 - v1
        y2 = y2 + Dx2 - v2

        # normOld = curNorm
        # curNorm = np.linalg.norm(Dx1 - v1) + np.linalg.norm(Dx2 - v2)
        #
        # if curNorm > alpha * normOld:
        #     rho = 0.5 * rho
        #
        rel_error = np.linalg.norm(x - x_old) / np.linalg.norm(x)
        if logging:
            print('%3g \t %3.5e' % (k, rel_error))

        if rel_error < tol:
            last_iter = k + 1
            break

    return x, last_iter


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    gt = plt.imread('House256.png')
    img = plt.imread('house.png')

    plt.subplot(1, 3, 1)
    plt.imshow(gt, cmap='gray')

    g = img * 255
    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap='gray')

    out, _ = admm(g, lam=15, rho=200, Nit=100, logging=True)

    plt.subplot(1, 3, 3)
    plt.imshow(np.array(out, dtype=np.int), cmap='gray')
    plt.show()

    print(PSNR(out / 255, gt))
