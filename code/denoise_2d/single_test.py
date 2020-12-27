from PySide6.QtWidgets import QApplication
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

from denoise_2d.tv2d_admm_matlab2 import admm
from drag_drop_view import DragDropView


def denoise(path):
    sigma = 50

    img = plt.imread(path)
    noisy_img = np.clip(img + (sigma / 255.0) * np.random.randn(*img.shape), 0, 1).astype('float32')

    start = time.time()
    out, last_iter = admm(noisy_img, lam=0.2, rho=4, Nit=1000, tol=1e-4, logging=True)
    out = np.clip(out, 0, 1).astype('float32')
    used_time = time.time() - start

    grid = np.column_stack([noisy_img, out, img])
    # plt.imsave(os.path.join(out_dir, name), grid, cmap='gray')
    plt.imshow(grid, cmap='gray')
    plt.show()

    psnr = PSNR(img, out)
    psnr_noisy = PSNR(img, noisy_img)
    ssim = SSIM(img, out, data_range=out.max() - out.min())
    ssim_noisy = SSIM(img, noisy_img, data_range=noisy_img.max() - noisy_img.min())

    print('Iter %d, Time %.4f' % (last_iter, used_time))
    print('Noisy PSNR: %.4f, PSNR %.4f, Noisy SSIM %.4f, SSIM %.4f' % (psnr_noisy, psnr, ssim_noisy, ssim))


if __name__ == '__main__':
    denoise('../testsets/test.png')
