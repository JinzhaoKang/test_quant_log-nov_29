import numpy as np
from scipy.stats import zscore


def binning(vals, bins: int, type: str, scale):
    var = np.var(vals)
    mean = np.mean(vals)
    if scale == 'squa':
        scale = mean
    if type == 'unif':
        a = mean - np.sqrt(3 * var)
        b = mean + np.sqrt(3 * var)
        partitions = np.linspace(a, b, num=2**bins + 1)
    elif type == 'gauss':
        if bins == 1:
            partitions = np.linspace(-1.596 * var + scale*mean, 1.596 * var + scale*mean, num=2**bins + 1)
        elif bins == 2:
            partitions = np.linspace(-1.991 * var + scale*mean, 1.992 * var + scale*mean, num=2**bins + 1)
        elif bins == 3:
            partitions = np.linspace(-2.344 * var + scale*mean, 2.345 * var + scale*mean, num=2**bins + 1)
        elif bins == 4:
            partitions = np.linspace(-2.68 * var + scale*mean, 2.69 * var + scale*mean, num=2**bins + 1)
#try not use only 1 var, like 2, 1/2
    return partitions


