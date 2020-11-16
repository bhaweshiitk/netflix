"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
from scipy.stats import multivariate_normal



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
    n, _ = X.shape
    K, _ = mixture.mu.shape
    p = mixture.p
    post = np.zeros((n, K))
    prob = np.zeros(n)
    for i in range(n):
        for j in range(K):
            post[i, j] = p[j] * multivariate_normal.pdf(X[i], mean=mixture.mu[j], cov=mixture.var[j])
        prob[i] = post[i, :].sum()
        post[i, :] = post[i, :] / prob[i]
    ll = np.sum(np.log(prob))
    return post, ll
    raise NotImplementedError


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
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    cost = 0
    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((mu[j] - X) ** 2).sum(axis=1) @ post[:, j]
        cost += sse
        var[j] = sse / (d * n_hat[j])

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
    prev_ll = None
    ll = None
    while (prev_ll is None or ((ll - prev_ll) >= abs((1e-6) * ll))):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, ll
    raise NotImplementedError
