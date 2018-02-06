import numpy as np

def jpt2cpt(jpt):
    max_y_xy = np.max(jpt, axis=1).reshape((-1, 1))
    x_margin = max_y_xy + np.log(np.sum(np.exp(jpt - max_y_xy),
                                        axis=1)).reshape((-1, 1))
    cpt = jpt - x_margin
    return cpt
