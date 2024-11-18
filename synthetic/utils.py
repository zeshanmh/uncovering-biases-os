import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy import stats


def pearsonr_ci(df, col_x, col_y, alpha=0.05):
    '''
        https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
    '''
    
    df_new = df[[col_x, col_y]].dropna()
    n = len(df_new) - 3

    r, p = stats.pearsonr(df_new[col_x], df_new[col_y])
    r_z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    lo, hi = np.tanh((lo_z, hi_z))

    return r, p, lo, hi


def rbf_kernel(X, sigma=1):
    X = np.atleast_2d(X)
    sq_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
    K = np.exp(-sq_dists / (2 * sigma ** 2))
    return K


def rbf_kernel_two_matrices(X1, X2, sigma=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    
    sq_dists = (
        np.sum(X1 ** 2, axis=1).reshape(-1, 1)  
        + np.sum(X2 ** 2, axis=1).reshape(1, -1)  
        - 2 * np.dot(X1, X2.T)  
    )
    
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K


def legendre_mat(X, d):
    matrix = np.zeros((X.shape[0], d))
    
    for n in range(d):
        Pn = Legendre([0] * (n + 1) + [1])
        matrix[:, n] = Pn(X)
    
    return matrix


def sample_bernoulli(row, covs, coefs=None, p=0.95):
    if coefs is not None:
        logit = row[covs] @ coefs[1:] + coefs[0]
        p = 1 / (1 + np.exp(-logit))

    return np.random.binomial(1, p)