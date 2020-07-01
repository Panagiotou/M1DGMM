# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:45:45 2020

@author: Utilisateur
"""

import os

os.chdir('C:/Users/rfuchs/Documents/GitHub/GLMLVM_MFA')

import autograd.numpy as np
from autograd.numpy.random import multivariate_normal, uniform, binomial

####################################################################
############# Univariate asta data generation ######################
####################################################################

def gen_z(numobs, seed):
    ''' Generate the latent variables to reproduce the asta paper results
    
    numobs (int): The number of observations desired
    seed (int): The random state seed 
    ----------------------------------------------------------------------
    returns (numobs 1darray): Univariate latent variable used in the asta paper example
    '''
    np.random.seed = seed

    true_w = [0.3, 0.4, 0.3]
    true_mu = [-1.26,0,1.26]
    true_cov = np.diag([0.055,0.0361,0.055])
    
    all_z = multivariate_normal(size = numobs, mean = true_mu, cov = true_cov)
    z = np.zeros(numobs)
    cases = uniform(0, 1, numobs)    
    labels = np.zeros(numobs)
    
    ranges = [0] + true_w + [1]
    
    for i in range(numobs):
        for dim in range(3):
            if (cases[i]>np.sum(ranges[:dim + 1])) and (cases[i]<np.sum(ranges[:dim + 2])):
                z[i] = all_z[i, dim]
                labels[i] = dim
    return z, labels


def gen_data(z, seed = None):
    ''' Generate a simulated dataset from univariate z
    
    z (numobs 1darray): Univariate latent variable
    seed (int): The random state seed 
    ----------------------------------------------------------------------
    returns (numobs x 4 ndarray): Synthetic dataset used in the asta paper example
    '''
    
    np.random.seed = seed

    numobs = len(z)  
    p1=2
    p2=1
    p3=1
    p=p1+p2+p3
    szo=5
    y = np.zeros(shape = (numobs, p))
    
    #===================================
    # Generate binary data
    #===================================
    
    alpha_zero = np.repeat([[0.50], [0.60]], axis = 1, repeats = numobs)
    #alphabin = np.repeat([[2], [2.50]], axis = 1, repeats = numobs)
    alphabin = np.array([[2], [2.50]])
    
    # A checker donne que des zeros...
    pred = alpha_zero + alphabin @ z[np.newaxis]
    probbin = np.exp(pred) / (1 + np.exp(pred))
    probbin = probbin.T
    casuali1 = uniform(size = (numobs, p1))
    x = np.zeros((numobs,p1))
    x = (casuali1 < probbin).astype(int)
    y[:,0:p1]=x
  
  
    #===================================
    # Generate count data
    #===================================  
    
    alpha_zcount = 0.70
    alphacount = [2.00]
  
    pred = alpha_zcount + alphacount @ z[np.newaxis]  
    probcount = np.exp(pred) / (1 + np.exp(pred))
  
    for l in range(numobs):
        y[l,p1:(p1+p2)] = binomial(n = 10, p = probcount[l], size = 1) # p1 + p2 ?
  
    #===================================
    # Generate ordinal data
    #===================================

    probor = np.zeros(shape = (numobs,szo,p3))
    thr = [-1.00, -0.30, 0.40,  0.90]
  
    alphaor = [[1.50]]
  
  
    for k in range(szo): # Truc chelou avec thr
        if (k<szo - 1):
            pred = thr[k] - alphaor @ z[np.newaxis]
            probor[:,k,:] = (np.exp(pred)/(1+ np.exp(pred))).T
        else:
            probor[:,k,:] = 1
            
    for l in range(numobs):                    
        for i in range(p3):
            x = uniform(0, 1, 1)[0]    
            for k in range(szo):
                if (k==0):
                    if (x<=probor[l,k,i]): 
                        y[l,p - 1] = k
                else:
                    if (probor[l,k-1,i]<x)&(x<=probor[l,k,i]):
                        y[l,p - 1] = k 
  
    return(y)


####################################################################
############## Multivariate data generation ########################
####################################################################

def gen_mv_z(numobs, r, init, seed):
    ''' Generate multivariate latent variable z from the initialisation parameters
    given in init.
    
    numobs (int): The number of observations desired
    r (int): The dimension of the latent variables
    init (dict): The initialisation parameters
    seed (int): The random state seed 
    ----------------------------------------------------------------------
    returns (numobs x r ndarray): Multivariate latent variables
    '''
    
    np.random.seed = seed

    k  = 3

    all_z = np.zeros((numobs,r,k))
    for i in range(k): # To change with np.block_diag when ready
        all_z[:,:,i] = multivariate_normal(size = numobs, mean = init['mu'][i].flatten(),\
                                           cov = init['sigma'][i]) 

    z = np.zeros((numobs, r))
    cases = uniform(0, 1, numobs)    
    labels = np.zeros(numobs)
    
    ranges = [0] + init['w'].flatten().tolist() + [1]
    
    for i in range(numobs):
        for dim in range(3):
            if (cases[i]>np.sum(ranges[:dim + 1])) and (cases[i]<np.sum(ranges[:dim + 2])):
                z[i] = all_z[i, : ,dim]
                labels[i] = dim
    return z, labels  

def gen_mvdata(z, init, seed = None):
    ''' Generate a simulated dataset from multivariate z
    
    z (numobs x r ndarray): Univariate latent variable
    init (dict): The initialisation parameters
    seed (int): The random state seed 
    ----------------------------------------------------------------------
    returns (numobs x 4 ndarray): A Synthetic dataset with 4 features
    '''
    
    np.random.seed = seed

    numobs = len(z)  
    p1 = 2
    p2 = 1
    p3 = 1
    p = p1+p2+p3
    max_nj_ord = 5
    y = np.zeros(shape = (numobs, p))
    
    r = init['lambda_bin'].shape[1] - 1
    
    #==============================================
    # Generate binary and count variables
    #==============================================
    
    pred = init['lambda_bin'] @ np.vstack([np.ones((1, numobs)), z.T]) 
    probbin = np.exp(pred) / (1 + np.exp(pred))
    probbin = probbin.T
    casuali1 = uniform(size = (numobs, p1))
    x = np.zeros((numobs,p1))
    x = (casuali1 < probbin[:,:p1]).astype(int)
    y[:,0:p1]=x

    # For count
    for l in range(numobs):
        y[l,p1:(p1+p2)] = binomial(n = 10, p = probbin[l,p1:(p1 + p2)], size = 1) # p1 + p2 ?

    #==============================================
    # Generate ordinal variable
    #==============================================

    probor = np.zeros(shape = (numobs, max_nj_ord, p3))
      
    lambda0 = init['lambda_ord'][:, :(max_nj_ord - 1)] # Shape (nb_ord, max_nj_ord)
    Lambda = init['lambda_ord'][:,(max_nj_ord - 1) :(max_nj_ord + r)] # Shape (nb_ord, r)
    
    for k in range(max_nj_ord): # Truc chelou avec thr
        if (k < max_nj_ord - 1):
            pred = lambda0[:,k] - Lambda @ z.T
            probor[:,k,:] = (np.exp(pred)/(1+ np.exp(pred))).T
        else:
            probor[:,k,:] = 1
            
    for l in range(numobs):                    
        for i in range(p3):
            x = uniform(0, 1, 1)[0]    
            for k in range(max_nj_ord):
                if (k==0):
                    if (x<=probor[l,k,i]): 
                        y[l,p - 1] = k
                else:
                    if (probor[l,k-1,i]<x)&(x<=probor[l,k,i]):
                        y[l,p - 1] = k 
  
    return(y)
    