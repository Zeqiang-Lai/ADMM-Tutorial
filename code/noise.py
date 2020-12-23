import numpy as np


def awgn_std(x, std):
    """
    additive white gaussian noise parameterized with std

    :param x: image normalized into 0-1
    :param std: std (0-255)
    :return: image with noise
    """
    std /= 255
    noise = np.random.normal(0, std, x.shape)
    return np.clip(x + noise, 0, 1), noise


def awgn_snr(x, snr):
    """
    Additive White Gaussian Noise

    :param x: input signal
    :param snr: signal noise ratio
    :return: signal with noise
    """
    Ps = np.mean(x ** 2)
    Pn = Ps / 10 ** (snr / 10)
    noise = np.random.randn(*x.shape) * np.sqrt(Pn)
    return x + noise, noise


def awgn_sd(x, snr):
    """ awgn signal dependent"""
    Ps = x
    Pn = Ps / (10 ** (snr / 10))
    noise = np.random.randn(*x.shape) * Pn
    return x + noise, noise