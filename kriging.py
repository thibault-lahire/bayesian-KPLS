#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:49:18 2020

@author: macbookthibaultlahire
"""

import numpy as np

import base as fbase
import correlations as correl



def surrogate_model(n, dim, fun, fun_base, theta, X, y, F):

    R, inv_R = correl.compute_R(theta, X)
    beta_hat = np.linalg.inv(F.T@inv_R@F)@F.T@inv_R@y
    sigma2_hat = (y - F@beta_hat).T@inv_R@(y - F@beta_hat)/n
            
    return R, inv_R, beta_hat, sigma2_hat


    
def pred(x, X, y, R, inv_R, F, beta_hat, sigma2_hat, theta):
    n, dim = X.shape
    f = fbase.quadratic(x)
    X_extended = np.vstack((x, X))
    D, _ = correl.l1_cross_distances(X_extended)
    corr = correl.squar_exp(theta, D)
    r = corr.T[0,:n].reshape(1, -1).T

    pred_ = f@beta_hat + r.T@inv_R@(y - F@beta_hat)
    var = sigma2_hat*(1 - r.T@inv_R@r)
    
    return pred_, var

