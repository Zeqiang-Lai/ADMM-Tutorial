import numpy as np


def PSNR(x, gt, data_range=1):
    mse = np.mean((x - gt) ** 2)
    return 10 * np.log10(data_range ** 2 / mse)


def forward_diff(u):
    dux = np.column_stack([np.diff(u, axis=1), u[:, 0] - u[:, -1]])
    duy = np.row_stack([np.diff(u, axis=0), u[0, :] - u[-1, :]])
    return dux, duy


def dive(x, y):
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=0)
    dtxy = np.column_stack([x[:, -1] - x[:, 0], -dx])
    dtxy = dtxy + np.row_stack([y[-1, :] - y[0, :], -dy])
    return dtxy


def defDDt():
    D = lambda u: forward_diff(u)
    Dt = lambda x, y: dive(x, y)
    return D, Dt


def shrink(x, r):
    return np.sign(x) * np.maximum(np.abs(x) - r, 0)


def admm(y, lam, rho, Nit, tol=1e-5):
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
    Dx1, Dx2 = D(x)

    curNorm = np.linalg.norm(Dx1 - v1) + \
              np.linalg.norm(Dx2 - v2)

    imgs = []

    for k in range(Nit):
        imgs.append(x)
        x_old = x
        ty1 = y1 / rho + v1
        ty2 = y2 / rho + v2
        tmp = Dt(ty1, ty2)
        rhs = y - rho * tmp
        lhs = 1 + rho * eigDtD

        x = np.fft.fft2(rhs) / lhs
        x = np.fft.ifft2(x).real

        Dx1, Dx2 = D(x)

        u1 = Dx1 + y1 / rho
        u2 = Dx2 + y2 / rho

        v1 = shrink(u1, lam / rho)
        v2 = shrink(u2, lam / rho)

        y1 = y1 + rho * (Dx1 - v1)
        y2 = y2 + rho * (Dx2 - v2)

        normOld = curNorm
        curNorm = np.linalg.norm(Dx1 - v1) + \
                  np.linalg.norm(Dx2 - v2)
        #
        # if curNorm > alpha * normOld:
        #     rho = 0.5 * rho
        #
        # relError = np.linalg.norm(x - x_old) / np.linalg.norm(x)
        # if relError < tol:
        #     break

    return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    gt = plt.imread('House256.png')
    img = plt.imread('house.png')

    plt.subplot(1, 3, 1)
    plt.imshow(gt, cmap='gray')

    g = img * 255
    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap='gray')

    out = admm(g, lam=29.7, rho=2, Nit=100)

    plt.subplot(1, 3, 3)
    plt.imshow(np.array(out, dtype=np.int), cmap='gray')
    plt.show()

    print(PSNR(out / 255, gt))
