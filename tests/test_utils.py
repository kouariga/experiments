import os

from experiments import utils
import numpy as np
from scipy import stats as stats
import pytest


@pytest.fixture
def jpt():
    # Define gaussian mixtures
    prior = [0.1, 0.6, 0.3]
    norms = [{'loc': -2, 'scale': 1, 'size': 1},
             {'loc': 0, 'scale': 1.5, 'size': 1},
             {'loc': 1, 'scale': 1.2, 'size': 1}]
    # Sample data
    n_samples = 10
    n_subs = stats.multinomial.rvs(n_samples, prior)
    xs = np.array([stats.norm.rvs(**norms[i])
                   for i, r in enumerate(n_subs)
                   for _ in range(r)]).reshape((-1, 1))

    # Calculate joint probabilities
    jpt = np.hstack([np.log(prior[i]) + stats.norm.logpdf(xs, loc=norm['loc'],
                                                          scale=norm['scale'])
                     for i, norm in enumerate(norms)]).reshape((n_samples, -1))
    return jpt


def test_jpt2cpt(jpt):
    cpt = utils.jpt2cpt(jpt)
    print(np.exp(cpt))
