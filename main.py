#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:50:54 2020

@author: macbookthibaultlahire
"""
#import os
import time

#os.chdir("C:/Users/samle/Documents/MAths/M1/MVA/Bayesian machine/mémoire/code")
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt

import base as fbase
import kriging as kg
import correlations as correl
import test_functions as tf
import DOE

plt.ion()
n = 13
dim = 1
fun = tf.sinus
fun_base = fbase.quadratic
theta = np.ones(dim)
X = DOE.lhs(dim, n)
y = fun(X)
F = fun_base(X)
R, inv_R, beta_hat, sigma2_hat = kg.surrogate_model(n, dim, fun, fun_base, theta, X, y, F)

# Find best theta

# Non bayesian manner
def logML(theta):
    F = fun_base(X)
    R, inv_R = correl.compute_R(theta, X)
    beta_hat = np.linalg.inv(F.T@inv_R@F)@F.T@inv_R@y
    sigma2_hat = (y - F@beta_hat).T@inv_R@(y - F@beta_hat)/n
    criterion = -0.5*(n*np.log(sigma2_hat) +np.log(np.linalg.det(R)))
    return -criterion



def constr1(theta):
    return theta

def constr2(theta):
    return -theta + 50  # lower bound can be changed






start = time.time()
new_theta = so.fmin_cobyla(logML, np.ones(dim), [constr1, constr2])
print("new_theta_MLE= ", new_theta)


#learning parameters
R, inv_R, beta_hat, sigma2_hat = kg.surrogate_model(n, dim, fun, fun_base, new_theta, X, y, F)
end = time.time()
print("Frequentist time =", (end-start))



test_set = DOE.lhs(dim, 50)*0.8+0.1 #we evaluate on the interval [0.1;0.9] to get rid of bound effects
true_value_on_test_set = fun(test_set)
norm_true = np.linalg.norm(true_value_on_test_set)
pred_value_on_test_set = np.zeros((50))
MSE = 0
for i in range(len(test_set)):
    #print(test_set[i])
    pred_value_on_test_set[i] = kg.pred([test_set[i]], X, y, R, inv_R, F, beta_hat, sigma2_hat, new_theta)[0]
    MSE += (pred_value_on_test_set[i] - true_value_on_test_set[i])**2
MSE = 100*np.sqrt(MSE)/norm_true
print("Frequentist MSE = ", MSE)



abscisse = np.linspace(0,1)
ord1 = fun(abscisse)
ord2 = np.zeros_like(abscisse)
ord3 = np.zeros_like(abscisse)
ord4 = np.zeros_like(abscisse)
for i in range(len(ord2)):
    res = kg.pred([[abscisse[i]]], X, y, R, inv_R, F, beta_hat, sigma2_hat, new_theta)
    pred_, var = res[0], res[1]
    ord2[i] = pred_[0,0]
    ord3[i] = pred_[0,0] + var[0,0]
    ord4[i] = pred_[0,0] - var[0,0]


plt.figure(1)
plt.clf()
plt.plot(abscisse, ord1, c='blue', label='true')
plt.plot(abscisse, ord2, c='red', label='guess')
plt.plot(abscisse, ord3, c='red', alpha=0.2)
plt.plot(abscisse, ord4, c='red', alpha=0.2)
plt.title('Kriging on a sinus, n={:}'.format(n))
plt.legend()
plt.savefig("kriging_on_sinus.png")
plt.show()


#optimisation ML
ab = np.linspace(0.5, 30)
ordonnee = np.zeros_like(ab)
for i in range(len(ab)):
    ordonnee[i] = logML(ab[i])
plt.figure(2)
plt.clf()
plt.plot(ab, ordonnee)
plt.title('- logML')
plt.xlabel('theta')

plt.show()


####     bayesian

#we learn from the frequentist MLE
#Inv gamma prior on sigma
delta=3#we take those parameter because they give good results empiricaly, they enable to underestimate the noise intensity and offer more regularity
alpha=15

# Gaussian prior on beta
b=beta_hat + 3*np.random.multivariate_normal(beta_hat.T[0], np.eye(3))

invGamma=10*np.eye(3)
#Gaussian prior on theta
th=new_theta + 3*np.random.normal(new_theta[0])
bias=new_theta.copy()
V=1000

def logbayes(theta):
    """(alpha,delta) parameter for the invGammma prior on sigma, 
    (b,invGamma) parameter for the Gaussian prior on beta, 
    (th,V) for the standard gaussian prior on (th,V>0)"""
    F = fun_base(X)
    R, inv_R = correl.compute_R(theta, X)
    beta_hat = np.linalg.inv(F.T@inv_R@F)@F.T@inv_R@y
    sigma2_hat = ((y - F@beta_hat).T@inv_R@(y - F@beta_hat)+2*delta)/(n+2*(alpha-1))#INVGAMA

    beta_hat = np.linalg.inv(F.T@inv_R@F/sigma2_hat**2+invGamma)@(invGamma@b+F.T@inv_R@y/sigma2_hat**2)
    More=0.5*(y - F@beta_hat).T@inv_R@(y - F@beta_hat)/sigma2_hat# we have to add this term because the expression of sigma2_hat does not imply further simplification as previouly whit the frequentist estimation of sigma2_hat
    criterion = -0.5*(n*np.log(sigma2_hat) + np.log(np.linalg.det(R))+np.linalg.norm(theta-th,2)**2/V)-More
    return -criterion


start = time.time()
new_theta = so.fmin_cobyla(logbayes, np.ones(dim), [constr1, constr2])
print("new_theta_bayes= ", new_theta)


#learning parameters
R, inv_R, beta_hat, sigma2_hat = kg.surrogate_model(n, dim, fun, fun_base, new_theta, X, y, F)
end = time.time()
print("Bayesian time =", (end-start))


pred_value_on_test_set = np.zeros((50))
MSE = 0
for i in range(len(test_set)):
    #print(test_set[i])
    pred_value_on_test_set[i] = kg.pred([test_set[i]], X, y, R, inv_R, F, beta_hat, sigma2_hat, new_theta)[0]
    MSE += (pred_value_on_test_set[i] - true_value_on_test_set[i])**2
MSE = 100*np.sqrt(MSE)/norm_true
print("Bayesian MSE = ", MSE)


abscisse = np.linspace(0,1)
ord1 = fun(abscisse)
ord2 = np.zeros_like(abscisse)
ord3 = np.zeros_like(abscisse)
ord4 = np.zeros_like(abscisse)
for i in range(len(ord2)):
    res = kg.pred([[abscisse[i]]], X, y, R, inv_R, F, beta_hat, sigma2_hat, new_theta)
    pred_, var = res[0], res[1]
    ord2[i] = pred_[0,0]
    ord3[i] = pred_[0,0] + var[0,0]
    ord4[i] = pred_[0,0] - var[0,0]


plt.figure(4)
plt.clf()
plt.plot(abscisse, ord1, c='blue', label='true')
plt.plot(abscisse, ord2, c='red', label='guess')
plt.plot(abscisse, ord3, c='red', alpha=0.2)
plt.plot(abscisse, ord4, c='red', alpha=0.2)
plt.title('Kriging on a sinus with bayesian learning, alpha={:},delta={:}'.format(alpha,delta))
plt.legend()
plt.show()


#optimisation ML
ab = np.linspace(0.5, 30)
ordonnee = np.zeros_like(ab)
for i in range(len(ab)):
    ordonnee[i] = logML(ab[i])
plt.figure(3)
plt.clf()
plt.plot(ab, ordonnee)
plt.title('- logML Bayes')
plt.xlabel('theta')

plt.show()



#sample posterior method, here sigma is supposed to be fixed and known


start = time.time()
#sigma2_hat = ((y - F@beta_hat).T@inv_R@(y - F@beta_hat)+2*beta)/(n+2*(alpha-1))#INVGAMA
sigma2_hat=(y - F@beta_hat).T@inv_R@(y - F@beta_hat)/n#sigma2_hat is fixed for sampling for sake of simplicity, there are more complicated technic to sample faithfully but we rely on basic methods here

#sigma2_hat = ((y - F@beta_hat).T@inv_R@(y - F@beta_hat)+2*beta)/(n+2*(alpha-1))#INVGAMA, we can take the bayesian estimator

beta=np.linalg.inv(F.T@inv_R@F)@F.T@inv_R@y# beta can be fixed

def distrib(theta):
    """ compute the density of theta|sigma,X, here we compute beta as a deterministic function of the other parameter"""
    F = fun_base(X)
    R, inv_R = correl.compute_R(theta, X)
    beta_hat = np.linalg.inv(F.T@inv_R@F)@F.T@inv_R@y# beta is computed, he is not supposed to be fixed here
    More= (y - F@beta_hat).T@inv_R@(y - F@beta_hat)/(2*sigma2_hat)
    P=np.exp(-(np.linalg.norm(theta-bias,2)**2)/(2*V)-More)/np.sqrt(np.linalg.det(R))
    return P



def distribFixe(theta):
    """ compute the density of theta|sigma,X,beta"""
    F = fun_base(X)
    R, inv_R = correl.compute_R(theta, X)
    More= (y - F@beta).T@inv_R@(y - F@beta)/(2*sigma2_hat)#beta is fixed here
    P=np.exp(-(np.linalg.norm(theta-bias,2)**2)/(2*V)-More)/np.sqrt(np.linalg.det(R))
    return P
def MCMC(F,theta0,sig,nburn,nsamp):
    """n burn time for burn, nsamp time for sampling, F proportional to the theta distribution to sample, sig footstep parameter, theta0 initial point """
    T=[theta0]
    d=len(theta0)
    theta=theta0
    k=0
    for i in range(nburn):
        theta1=np.random.multivariate_normal(theta,sig*np.eye(d))
        u=np.random.random()
        if u<F(theta1)/F(theta):
            theta=theta1
            k=k+1
    T.append(theta)
    for i in range(nsamp):
        theta1=np.random.multivariate_normal(theta,sig*np.eye(d))
        u=np.random.random()
        if u<F(theta1)/F(theta):
            theta=theta1
            k=k+1
            T.append(theta)
    print("Number of MCMC steps :", k)
    T=np.array(T)
    return T

#sampling visualisation
plt.figure(5)
plt.clf()

sig=0.1#footstep parameter
T=MCMC(distrib,bias,sig,2000,7000)#we estimate beta at each step
plt.title("sampling of theta according to the posterior, n={:}".format(n))
plt.hist(T,range = (0, 50), bins = 1000)
plt.show()
plt.ion()

plt.figure(9)
plt.clf()


T=MCMC(distribFixe,bias,sig,2000,7000)#we fixe beta
plt.title("sampling of theta according to the posterior, n={:}".format(n))
plt.hist(T,range = (0, 50), bins = 1000)
plt.show()
plt.ion()

new_theta=np.sum(T)/len(T)#we take the mean as our MP estimator
print("new_theta_from_posterior= ", new_theta)


#learning parameters
R, inv_R, beta_hat, sigma2_hat = kg.surrogate_model(n, dim, fun, fun_base, new_theta, X, y, F)
end = time.time()
print("theta by 'Sampling from posterior' time =", (end-start))


pred_value_on_test_set = np.zeros((50))
MSE = 0
for i in range(len(test_set)):
    #print(test_set[i])
    pred_value_on_test_set[i] = kg.pred([test_set[i]], X, y, R, inv_R, F, beta_hat, sigma2_hat, new_theta)[0]
    MSE += (pred_value_on_test_set[i] - true_value_on_test_set[i])**2
MSE = 100*np.sqrt(MSE)/norm_true
print("After posterior sampling MSE = ", MSE)


abscisse = np.linspace(0,1)

ord1 = fun(abscisse)
ord2 = np.zeros_like(abscisse)
ord3 = np.zeros_like(abscisse)
ord4 = np.zeros_like(abscisse)
for i in range(len(ord2)):
    res = kg.pred([[abscisse[i]]], X, y, R, inv_R, F, beta_hat, sigma2_hat, new_theta)
    pred_, var = res[0], res[1]
    ord2[i] = pred_[0,0]
    ord3[i] = pred_[0,0] + var[0,0]
    ord4[i] = pred_[0,0] - var[0,0]


plt.figure(6)
plt.clf()
plt.plot(abscisse, ord1, c='blue', label='true')
plt.plot(abscisse, ord2, c='red', label='guess')
plt.plot(abscisse, ord3, c='red', alpha=0.2)
plt.plot(abscisse, ord4, c='red', alpha=0.2)
plt.title('Kriging with mean of the posterior of theta, n={:}'.format(n))
plt.legend()
plt.show()




