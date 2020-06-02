import numpy as np


def cfar2d(x, num_train, num_guard, rate_fa):
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        out[:, c] = cfar(x[:, c], num_train, num_guard, rate_fa).T

    return out


def cfar(x, num_train, num_guard, rate_fa):
    """
    Detect peaks with CFAR algorithm.

    num_train: Number of training cells.
    num_guard: Number of guard cells.
    rate_fa: False alarm rate.
    """
    num_train_half = round(num_train / 2)
    num_guard_half = round(num_guard / 2)
    num_side = num_train_half + num_guard_half

    out = np.zeros_like(x)

    alpha = num_train * (rate_fa**(-1 / num_train) - 1)  # threshold factor

    # generate mask
    mask = np.ones(num_side * 2)
    mask[num_train_half:num_guard] = 0
    mask /= num_train

    noise = np.convolve(x, mask, 'same')

    threshold = alpha * noise
    out = np.greater(x, threshold) * 255

    return out
