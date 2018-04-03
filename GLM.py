#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:19:50 2018

@author: clytie
"""

import numpy as np
#from numpy.linalg import LinAlgError


class NewtonMethod(object):
    def __init__(self, niters = 100, intercept = True, tol = 1e-5):
        self.niters = niters
        self.tol = tol
        self.intercept = intercept
    
    def _derivative(self, X, y, beta):
        raise NotImplementedError
        
    def _Hessian(self, X, beta):
        raise NotImplementedError
        
    def __str__(self):
        return ("%s(niters = %s, intercept = %s, tol = %s)"
                              % (self.__class__.__name__, self.niters, self.intercept, self.tol))    
    
    def fit(self, X, y, beta_init = None, verbose = 1):
        self.n, p = X.shape
        if self.intercept:
            self.p = p + 1
            beta = np.zeros(self.p).reshape(-1, 1)
            X = np.append(np.ones((self.n, 1)), X, axis = 1)
        else:
            self.p = p 
            beta = np.zeros(self.p).reshape(-1, 1)
        
        if not beta_init is None:
            for i in np.arange(len(beta_init)):
                beta[i] = beta_init[i]
                
        niter = 1
        tol = np.inf
        while 1:
            g = self._derivative(X, y, beta)
            H = self._Hessian(X, beta)
            beta_new = beta - np.dot(np.linalg.inv(H), g)
#            try:
#                beta_new = beta - np.dot(np.linalg.inv(H), g)
#            except LinAlgError:
#                print("Singular matrix!")

            if verbose:
                print("iteration %s:\n %s" % (niter, str(beta_new.reshape(-1))))
        
            tol = np.sqrt(np.sum((beta_new - beta) ** 2))
            if tol < self.tol or niter >= self.niters:
                self.coefs_ = beta_new
                break
            else:
                niter += 1
                beta = beta_new
           
        self.niter = niter
        
        print("\n")
        if self.niter < self.niters:
            print("Converge!")
        else:
            print("Not converge!")
        print("iteration %s: %s" % (niter, str(self.coefs_.reshape(-1))))


class Logistic(NewtonMethod):
    def __init__(self, niters = 100, intercept = True, tol = 1e-5):
        super(Logistic, self).__init__(
                niters = niters, intercept = intercept, tol = tol)

    def _logit(self, xi, beta):
        #注意这里如果不变成一维向量会出错
        xi = xi.reshape(-1)
        beta = beta.reshape(-1)
        return 1. / (1 + np.exp(- np.sum(xi * beta)))
    
    def _derivative(self, X, y, beta):
        gradient = np.zeros((self.p, 1))
        for i in np.arange(self.n):
            pi = self._logit(X[i], beta)
            gradient += (pi - y[i]) * X[i].reshape(-1, 1)
        return gradient / self.n
    
    def _Hessian(self, X, beta):
        Hessian = np.zeros((self.p, self.p))
        for i in np.arange(self.n):
            pi = self._logit(X[i], beta)
            Hessian += pi * (1 - pi) * np.dot(X[i].reshape(-1, 1), X[i].reshape(1, -1))
        return Hessian / self.n        


class Poisson(NewtonMethod):
    def __init__(self, niters = 100, intercept = True, tol = 1e-5):
        super(Poisson, self).__init__(
                niters = niters, intercept = intercept, tol = tol)
    
    def _log(self, xi, beta):
        xi = xi.reshape(-1)
        beta = beta.reshape(-1)
        return np.exp(np.sum(xi * beta))
    
    def _derivative(self, X, y, beta):
        gradient = np.zeros((self.p, 1))
        for i in np.arange(self.n):
            lambdai = self._log(X[i], beta)
            gradient += (lambdai - y[i]) * X[i].reshape(-1, 1)
        return gradient

    def _Hessian(self, X, beta):
        Hessian = np.zeros((self.p, self.p))
        for i in np.arange(self.n):
            lambdai = self._log(X[i], beta)
            Hessian += lambdai * np.dot(X[i].reshape(-1, 1), X[i].reshape(1, -1))
        return Hessian            
        
"""Logistic模拟"""
#模拟数据
n, p = 1000, 10
beta = np.arange(1, p + 2)
X = np.random.normal(size = (n, p))           
#epsilon = np.random.normal(size = (n))    
X_intercept = np.append(np.ones((n, 1)), X, axis = 1)
#yita = np.dot(X_intercept, beta) + epsilon
yita = np.dot(X_intercept, beta)
p = 1. / (1 + np.exp(- yita))

y = np.array([np.random.binomial(1, pi) for pi in p])
            
#拟合模型
lr = Logistic()
lr.fit(X, y)
            

"""Poisson模拟"""
#模拟数据
n, p = 1000, 10
beta = np.arange(1, p + 2) / 10.
X = np.random.normal(size = (n, p))           
#epsilon = np.random.normal(size = (n))    
X_intercept = np.append(np.ones((n, 1)), X, axis = 1)
#yita = np.dot(X_intercept, beta) + epsilon
yita = np.dot(X_intercept, beta)
lambdas = np.exp(yita)

y = np.array([np.random.poisson(lambdai) for lambdai in lambdas])

#拟合模型
ps = Poisson()            
ps.fit(X, y)            
            
            