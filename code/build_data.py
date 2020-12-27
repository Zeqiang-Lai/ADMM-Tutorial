import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    dataroot = 'BSD68'
    std = 15
    out_dir = 'BSD68-std' + str(std)
    os.makedirs(out_dir, exist_ok=True)

    std = float(std)

    names = os.listdir(dataroot)

    for idx, name in enumerate(names):
        if not name.endswith('png'):
            break
        print('Process (%d|%d): ' % (idx + 1, len(names)) + name)
        path = os.path.join(dataroot, name)
        img = plt.imread(path)
        sigma = std / 255
        noisy_img = np.clip(img + sigma * np.random.randn(*img.shape), 0, 1).astype('float32')

        plt.imsave(os.path.join(out_dir, name), noisy_img, cmap='gray')

        break
