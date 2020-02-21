#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:28:17 2020

@author: macbookthibaultlahire
"""

import numpy as np
import math


def l1_cross_distances(X):

    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.
    Parameters
    ----------
    X: np.ndarray [n_obs, dim]
            - The input variables.
    Returns
    -------
    D: np.ndarray [n_obs * (n_obs - 1) / 2, dim]
            - The L1 cross-distances between the vectors in X.
    ij: np.ndarray [n_obs * (n_obs - 1) / 2, 2]
            - The indices i and j of the vectors in X associated to the cross-
              distances in D.
    """

    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0

    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1) : n_samples])

    return D, ij.astype(np.int)



def squar_exp(theta, d):

    """
    Squared exponential correlation model.
    Parameters
    ----------
    theta : list[ncomp]
        the autocorrelation parameter(s).
    d: np.ndarray[n_obs * (n_obs - 1) / 2, n_comp]
        |d_i * coeff_pls_i| if PLS is used, |d_i| otherwise
    Returns
    -------
    r: np.ndarray[n_obs * (n_obs - 1) / 2,1]
        An array containing the values of the autocorrelation model.
    """

    r = np.zeros((d.shape[0], 1))
    n_components = d.shape[1]

    # Construct/split the correlation matrix
    i, nb_limit = 0, int(1e4)

    while True:
        if i * nb_limit > d.shape[0]:
            return r
        else:
            r[i * nb_limit : (i + 1) * nb_limit, 0] = np.exp(
                -np.sum(
                    theta.reshape(1, n_components)
                    * d[i * nb_limit : (i + 1) * nb_limit, :],
                    axis=1,
                )
            )
            i += 1




def compute_R(theta, X):
    n_samples, dim = X.shape
    D, idx = l1_cross_distances(X)
    corr = squar_exp(theta, D)
    R = np.eye(n_samples)
    for i in range(corr.shape[0]):
        R[idx[i][0],idx[i][1]] = corr[i][0]
    R = R + R.T - np.eye(n_samples)
    inv_R = np.linalg.inv(R)
    return R, inv_R


if __name__ == '__main__':
    x1 = np.ones(10)
    x2 = np.ones(10)+2
    x3 = np.ones(10)+3
    x4 = np.ones(10)+0.8
    X = np.vstack((x1, x2, x3, x4))
    print(X.shape)
    theta = np.ones(10)*0.1
    res = r(x1, x2, theta)
    print(res)
    res2, _ = l1_cross_distances(X)
    print(res2)
    print()
    res3 = squar_exp(theta, res2)
    print("final res = ", res3)
    print("ATTENTION", _)
    print("ATTENTION", _+1)
    idx = _
    matrix_R = np.eye(4)
    for i in range(res3.shape[0]):
        print("achtung", idx[i][1])
        print("achtung", res3[i][0])
        matrix_R[idx[i][0],idx[i][1]] = res3[i][0]
    print("FINAL", matrix_R)
    matrix_R = matrix_R + matrix_R.T - np.eye(4)
    print(matrix_R)
    
#    for i in range(1, matrix_R.shape[0]):
#        for j in range(i):
#            matrix_R[i,j] = 0#res3[i+j,0]
    

