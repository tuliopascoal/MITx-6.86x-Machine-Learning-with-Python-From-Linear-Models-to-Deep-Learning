"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n = X.shape[0]
    K = mixture.mu.shape[0]
    post = np.zeros((n, K))

    l_old = 0
    for i in range(n):
        for j in range(K):
            likelihood = gaussian(X[i], mixture.mu[j], mixture.var[j])
            post[i, j] = mixture.p[j] * likelihood
        total_sum = post[i, :].sum()
        post[i, :] = post[i, :] / total_sum
        l_old += np.log(total_sum)

    return post, l_old
    raise NotImplementedError

def gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the probablity of vector x under a normal distribution

    Args:
        x: (d, ) array holding the vector's coordinates
        mean: (d, ) mean of the gaussian
        var: variance of the gaussian

    Returns:
        float: the probability
    """
    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean)**2).sum() / var
    return np.exp(log_prob)

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    nhat = post.sum(axis=0)
    p = nhat / n

    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        # Computing mean
        mu[j, :] = (X * post[:, j, None]).sum(axis=0) / nhat[j]
        # Computing variance
        sse = ((mu[j] - X) ** 2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * nhat[j])

    return GaussianMixture(mu, var, p)
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    l_old = None
    ll = None
    while (l_old is None or ll - l_old > 1e-6 * np.abs(ll)):
        l_old = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, ll
    raise NotImplementedError
