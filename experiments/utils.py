import numpy as np


def x_marginal(jpt):
    max_y_xy = np.max(jpt, axis=1).reshape((-1, 1))
    x_margin = max_y_xy + np.log(np.sum(np.exp(jpt - max_y_xy),
                                          axis=1)).reshape((-1, 1))
    return x_margin


def jpt2cpt(jpt):
    x_margin = x_marginal(jpt)
    return (jpt - x_margin)


def entropy(jpt):
    cpt = jpt2cpt(jpt)
    h = - np.sum(np.multiply(np.exp(cpt), cpt), axis=1)
    return h


def weighted_entropy(jpt):
    h = entropy(jpt)
    x_margin = x_marginal(jpt).ravel()
    wh = np.multiply(np.exp(x_margin), h)
    return wh
