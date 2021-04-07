# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:55:47 2021

@author: rfuchs
"""

import autograd.numpy as np
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t
from scipy.special import binom
from sklearn.preprocessing import OneHotEncoder
from numeric_stability import log_1plusexp, expit, softmax_



'''
lambda_bin = lambda_bin
z_new = z_

'''

def draw_new_bin(lambda_bin, z_new, nj_bin): 
    ''' A Adapter
    
    Generates draws from p(y_j | zM, s1 = k1) of the binary/count variables
    
    lambda_bin ( (r + 1) 1darray): Coefficients of the binomial distributions in the GLLVM layer
    z_new (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    nj_bin_j: ...
    --------------------------------------------------------------
    returns (ndarray): p(y_j | zM, s1 = k1)
    '''

    new_nb_obs = z_new.shape[0]
    r = z_new.shape[1]
    
    # Draw one y per new z for the moment
    nb_bin = len(nj_bin)
    y_bin_new = np.full((new_nb_obs, nb_bin), np.nan)
    
    for j in range(nb_bin):
    
        # Compute the probability
        eta = z_new @ lambda_bin[j][1:][..., n_axis]
        eta = eta + lambda_bin[j][0].reshape(1, 1) # Add the constant
        pi = expit(eta)
        
        # Draw the observations
        u = np.random.uniform(size = (new_nb_obs, nj_bin[j])) # To check: work for binomials
        y_bin_new[:,j] = (u > pi).sum(1)          
        
    return y_bin_new


def draw_new_categ(lambda_categ, z_new, nj_categ):
    ''' A Adapter
    Generates draws from p(y_j | zM, s1 = k1) of the categorical variables
    
    lambda_categ (nj_categ x (r + 1) ndarray): Coefficients of the categorical distributions in the GLLVM layer
    z_new (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_categ_j (int): The number of possible values values of the jth categorical variable
    --------------------------------------------------------------
    returns (ndarray): The p(y_j | zM, s1 = k1) for the jth categorical variable
    '''  
    epsilon = 1E-10
    
    nb_categ = len(nj_categ)
    new_nb_obs = z_new.shape[0]
    r = z_new.shape[1] 
    
    y_categ_new = np.full((new_nb_obs, nb_categ), np.nan)
  
    for j in range(nb_categ):
        
        zM_broad = np.expand_dims(np.expand_dims(z_new, 1), 2)
        lambda_categ_j_ = lambda_categ[j].reshape(nj_categ[j], r + 1, order = 'C')

        # Compute the probability
        eta = zM_broad @ lambda_categ_j_[:, 1:][n_axis,..., n_axis] # Check que l'on fait r et pas k ?
        eta = eta + lambda_categ_j_[:,0].reshape(1, nj_categ[j], 1, 1) # Add the constant
        pi = softmax_(eta.astype(np.float), axis = 1) 
        
        # Numeric stability
        pi = np.where(pi <= 0, epsilon, pi)
        pi = np.where(pi >= 1, 1 - epsilon, pi)
        
        # Draw the observations
        pi = pi[:,:,0,0]
        cumsum_pi = np.cumsum(pi, axis = 1)
        u = np.random.uniform(size = (new_nb_obs, 1)) # To check: work for binomials
        y_categ_new[:,j] = (cumsum_pi > u).argmax(1)  

    return y_categ_new


def draw_new_ord(lambda_ord, z_new, nj_ord): 
    '''  A adapter
    Generates draws from p(y_j | zM, s1 = k1) of the ordinal variables

    lambda_ord ( (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    z_new (... x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    nj_ord (int): The number of possible values values of the jth ordinal variable
    --------------------------------------------------------------
    returns (ndarray): The p(y_j | zM, s1 = k1) for the jth ordinal variable
    '''    
    r = z_new.shape[1]
    epsilon = 1E-10 # Numeric stability
    nb_ord = len(nj_ord)
    new_nb_obs = z_new.shape[0]

    y_ord_new = np.full((new_nb_obs, nb_ord), np.nan)

    for j in range(nb_ord):
        
        lambda0 = lambda_ord[j][:(nj_ord[j] - 1)]
        Lambda = lambda_ord[j][-r:]
 
        broad_lambda0 = lambda0.reshape((1, nj_ord[j] - 1))
        eta = broad_lambda0 - (z_new @ Lambda.reshape((r, 1)))
        
            
        gamma = expit(eta)
        gamma_prev = np.concatenate([np.zeros((new_nb_obs, 1)), gamma], axis = 1)
        
        # Draw the observations
        u = np.random.uniform(size = (new_nb_obs, 1)) # To check: work for binomials
        y_ord_new[:,j] = (gamma_prev > u).argmax(1)  
           
    return y_ord_new


def draw_new_cont(lambda_cont, z_new):
    '''  A adapter 
    Generates draws from p(y_j | zM, s1 = k1) of the continuous variables
    
    y_cont_j (numobs 1darray): The subset containing only the continuous variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    --------------------------------------------------------------
    returns (ndarray): p(y_j | zM, s1 = k1)
    '''
    
    r = z_new.shape[1]
    nb_cont = lambda_cont.shape[0]
    new_nb_obs = z_new.shape[0]

    y_cont_new = np.full((new_nb_obs, nb_cont), np.nan)
    
    for j in range(nb_cont):
        eta = z_new @ lambda_cont[j][1:].reshape(r, 1)
        eta = eta + lambda_cont[j][0].reshape(1, 1) # Add the constant
        
        y_cont_new[:,j] = np.random.multivariate_normal(mean = eta.flatten(),\
                                                    cov = np.eye(new_nb_obs))
        
    return y_cont_new


