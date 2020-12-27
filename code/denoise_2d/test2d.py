from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

import os
import matplotlib.pyplot as plt
import numpy as np
import json
import time

from tv2d_admm_matlab2 import admm

if __name__ == '__main__':
    dataroot = 'BSD68'
    debug = True
    names = os.listdir(dataroot)

    results = {}

    avg_psnr = 0
    avg_ssim = 0
    avg_iter = 0
    avg_time = 0

    avg_noisy_psnr = 0
    avg_noisy_ssim = 0

    sigma = 50
    out_dir = 'result' + str(sigma)
    os.makedirs(out_dir, exist_ok=True)

    for idx, name in enumerate(names):
        if not name.endswith('png'):
            break
        print('Process (%d|%d): ' % (idx + 1, len(names)) + name)
        path = os.path.join(dataroot, name)
        img = plt.imread(path)
        noisy_img = np.clip(img + (sigma / 255.0) * np.random.randn(*img.shape), 0, 1).astype('float32')

        start = time.time()
        out, last_iter = admm(noisy_img, lam=0.015, rho=4, Nit=1000, tol=1e-4, logging=debug)
        out = np.clip(out, 0, 1).astype('float32')
        used_time = time.time() - start

        grid = np.column_stack([noisy_img, out, img])
        plt.imsave(os.path.join(out_dir, name), grid, cmap='gray')
        if debug:
            plt.imshow(grid, cmap='gray')
            plt.show()

        psnr = PSNR(img, out)
        psnr_noisy = PSNR(img, noisy_img)
        ssim = SSIM(img, out, data_range=out.max() - out.min())
        ssim_noisy = SSIM(img, noisy_img, data_range=noisy_img.max() - noisy_img.min())

        print('Iter %d, Time %.4f' % (last_iter, used_time))
        print('Noisy PSNR: %.4f, PSNR %.4f, Noisy SSIM %.4f, SSIM %.4f' % (psnr_noisy, psnr, ssim_noisy, ssim))

        results[name] = {'psnr': psnr, 'ssim': ssim, 'ssim_noisy': ssim_noisy, 'psnr_noisy': psnr_noisy,
                         'iter': last_iter, 'time': used_time}

        avg_psnr += psnr
        avg_ssim += ssim
        avg_iter += last_iter
        avg_time += used_time
        avg_noisy_psnr += psnr_noisy
        avg_noisy_ssim += ssim_noisy

        if debug:
            break

    avg_psnr /= len(names)
    avg_ssim /= len(names)
    avg_iter /= len(names)
    avg_time /= len(names)
    avg_noisy_psnr /= len(names)
    avg_noisy_ssim /= len(names)

    print('Average Iter: %.4f, Average Time: %.4f' % (avg_iter, avg_time))
    print('Average PSNR: %.4f, Average SSIM: %.4f' % (avg_psnr, avg_ssim))
    print('Average Noisy PSNR: %.4f, Average Noisy SSIM: %.4f' % (avg_noisy_psnr, avg_noisy_ssim))

    with open(dataroot+'-2d-' + str(sigma) + 'json', 'w') as f:
        json.dump({'results': results, 'avg_psnr': avg_psnr, 'avg_ssim': avg_ssim}, f)
