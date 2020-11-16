"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from scipy.stats import multivariate_normal


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
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
    ll = 0
    logsum1 = 0
    for i in range(n):
        for j in range(K):
            post[i, j] = np.log(p[j]+1e-16) - ((np.linalg.norm((X[i][X[i] != 0] - mixture.mu[j][X[i] != 0]))) ** 2) / (
                        2 * (mixture.var[j]))-0.5 * len(X[i][X[i] != 0]) * np.log((2 * np.pi * (mixture.var[j]) + 1e-16))
        logsum1 = logsumexp(post[i])
        ll = ll + logsum1
        for j in range(K):
            post[i, j] = post[i, j] - logsum1
    return np.exp(post), ll
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    cost = 0
    mu = mixture.mu
    var = np.zeros(K)
    sse = 0
    dim = 0
    for j in range(K):
        for i in range(d):
            prob_sum = np.sum(post[:, j][X[:, i] != 0])
            if prob_sum >= 1:
                mu[j, i] = np.sum(np.multiply(post[:, j], X[:, i])) / prob_sum

    for j in range(K):
        sse = 0
        dim = 0
        for i in range(n):
            sse = sse + (np.linalg.norm((X[i, :][X[i, :] != 0] - mu[j, :][X[i, :] != 0])) ** 2) * post[i, j]
            dim = dim + len(X[i, :][X[i, :] != 0]) * post[i, j]
        var[j] = max(min_variance, sse / dim)
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
        mixture = mstep(X,  post, mixture)

    return mixture, post, ll
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X1 = np.copy(X)
    n, d = X.shape
    K, _ = mixture.mu.shape
    p = mixture.p
    post = np.zeros((n, K))
    ll = 0
    logsum1 = 0
    for i in range(n):
        for j in range(K):
            post[i, j] = np.log(p[j] + 1e-16) - (
                        (np.linalg.norm((X[i][X[i] != 0] - mixture.mu[j][X[i] != 0]))) ** 2) / (
                                 2 * (mixture.var[j])) - 0.5 * len(X[i][X[i] != 0]) * np.log(
                (2 * np.pi * (mixture.var[j]) + 1e-16))
        logsum1 = logsumexp(post[i])
        ll = ll + logsum1
        for j in range(K):
            post[i, j] = post[i, j] - logsum1
    post = np.exp(post)
    for i in range(n):
        for j in range(d):
            if X1[i][j] == 0:
                X1[i][j] = np.sum(np.multiply(mixture.mu[:, j], post[i, :]))
    return X1
    raise NotImplementedError
