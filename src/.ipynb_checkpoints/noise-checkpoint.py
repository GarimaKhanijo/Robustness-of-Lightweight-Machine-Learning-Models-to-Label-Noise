import numpy as np


def add_symmetric_noise(y, noise_rate, random_state=42):
    """
    Apply symmetric (uniform) label noise.
    
    noise_rate: float (e.g., 0.1 for 10%)
    Randomly flips labels with given probability.
    """

    np.random.seed(random_state)
    y_noisy = y.copy()

    n_samples = len(y)
    n_noisy = int(noise_rate * n_samples)

    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    for idx in noisy_indices:
        y_noisy[idx] = 1 - y_noisy[idx]  # binary flip

    return y_noisy


def add_asymmetric_noise(y, noise_rate, random_state=42):
    """
    Apply asymmetric label noise:
    Flip class 1 -> 0 only.
    
    noise_rate: fraction of class 1 samples to flip.
    """

    np.random.seed(random_state)
    y_noisy = y.copy()

    class_1_indices = np.where(y == 1)[0]
    n_class_1 = len(class_1_indices)

    n_noisy = int(noise_rate * n_class_1)

    noisy_indices = np.random.choice(class_1_indices, n_noisy, replace=False)

    for idx in noisy_indices:
        y_noisy[idx] = 0  # flip 1 -> 0

    return y_noisy