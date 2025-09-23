import math
import numpy as np

def our_mean(values):
    n = len(values)
    return sum(values) / n if n > 0 else float('nan')

def our_variance(values):
    n = len(values)
    m = our_mean(values)
    return sum((x - m) ** 2 for x in values) / n if n > 0 else float('nan')

def our_std(values):
    return math.sqrt(our_variance(values))

def our_percentile(values, p):
    n = len(values)
    if n == 0:
        return float('nan')
    values_sorted = sorted(values)
    k = (n - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values_sorted[int(f)] * (c - k)
    d1 = values_sorted[int(c)] * (k - f)
    return d0 + d1



def sigmoid(z):
    return 1 / (1 + np.exp(-z))
