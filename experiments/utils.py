import numpy as np
import scipy as sp


def y_margin_p(jpt):
    return sp.misc.logsumexp(jpt, axis=0, keepdims=True)


def x_margin_p(jpt):
    return sp.misc.logsumexp(jpt, axis=1, keepdims=True)


def jpt_to_ycpt(jpt):
    return (jpt - y_margin_p(jpt))
    

def jpt_to_xcpt(jpt):
    return (jpt - x_margin_p(jpt))


def gibbs(entropy, beta, log=True):
    print(entropy)
    if log:
        gibbs = entropy**(- beta)/np.sum(entropy**(- beta))
    else:
        gibbs = (np.exp(- beta*entropy)/np.sum(np.exp(- beta*entropy)))
    return gibbs.ravel()


def uncertainty(jpt, beta=10):
    xcpt = jpt_to_xcpt(jpt)
    entropy = - np.sum(np.multiply(np.exp(xcpt), xcpt), axis=1)
    uncertainty = gibbs(np.log(entropy), beta)
    return uncertainty


def likelihood(jpt, beta=10):
    ycpt = jpt_to_ycpt(jpt)
    entropy = - np.max(ycpt, axis=1)
    likelihood = gibbs(entropy, beta)
    return likelihood
 
 
# def weighted_entropy(jpt):
#     h = entropy(jpt)
#     x_margin = x_marginal(jpt).ravel()
#     wh = np.multiply(np.exp(x_margin), h)
#     return wh
