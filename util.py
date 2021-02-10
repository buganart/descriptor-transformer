import numpy as np


def remove_outliers(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return x[(np.abs(x - mean) < std * 5).all(axis=1), :]
