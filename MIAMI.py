# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:11:28 2021

@author: RobF
"""

import pandas as pd
from m1dgmm import M1DGMM
from scipy.linalg import block_diag
from oversample import draw_new_bin, draw_new_ord,\
                            draw_new_categ, draw_new_cont
                            
import autograd.numpy as np
from autograd.numpy.random import multivariate_normal


def MIAMI(y, n_clusters, r, k, init, var_distrib, nj, authorized_ranges,\
          target_nb_pseudo_obs = 500, it = 50, \
          eps = 1E-05, maxstep = 100, seed = None, perform_selec = True,\
              dm = [], max_patience = 1): # dm: Hack to remove
    
    ''' Generates pseudo-observations from a trained M1DGMM
    
    y (numobs x p ndarray): The observations containing mixed variables
    n_clusters (int): The number of clusters to look for in the data
    r (list): The dimension of latent variables through the first 2 layers
    k (list): The number of components of the latent Gaussian mixture layers
    init (dict): The initialisation parameters for the algorithm
    var_distrib (p 1darray): An array containing the types of the variables in y 
    nj (p 1darray): For binary/count data: The maximum values that the variable can take. 
                    For ordinal data: the number of different existing categories for each variable
    authorized_ranges (ndarray): The ranges in which the observations have to lie in
    target_nb_pseudo_obs (int): The number of pseudo-observations to generate         
    it (int): The maximum number of MCEM iterations of the algorithm
    eps (float): If the likelihood increase by less than eps then the algorithm stops
    maxstep (int): The maximum number of optimisation step for each variable
    seed (int): The random state seed to set (Only for numpy generated data for the moment)
    perform_selec (Bool): Whether to perform architecture selection or not
    dm (np array): The distance matrix of the observations. If not given M1DGMM computes it
    ------------------------------------------------------------------------------------------------
    returns (dict): The predicted classes, the likelihood through the EM steps
                    and a continuous representation of the data
    '''

    out = M1DGMM(y, 'auto', r, k, init, var_distrib, nj, it,\
             eps, maxstep, seed, perform_selec = perform_selec,\
                 dm = dm, max_patience = max_patience)
        
    # Upacking the model from the M1DGMM output
    #best_z = out['best_z']
    k = out['best_k']
    r = out['best_r']
    w_s = out['best_w_s'] 
    lambda_bin = out['lambda_bin'] 
    lambda_ord = out['lambda_ord'] 
    lambda_categ = out['lambda_categ'] 
    lambda_cont = out['lambda_cont'] 
    mu_s = out['mu'] 
    sigma_s = out['sigma'] 
    
    nj_bin = nj[pd.Series(var_distrib).isin(['bernoulli', 'binomial'])].astype(int)
    nj_ord = nj[var_distrib == 'ordinal'].astype(int)
    nj_categ = nj[var_distrib == 'categorical'].astype(int)

    y_std = y[:,var_distrib == 'continuous'].std(axis = 0, keepdims = True)
    
    M0 = 100 # The number of z to draw 
    S0 = np.prod(k)
    MM = 30 # The number of y to draw for each z
        
    #=======================================================
    # Data augmentation part
    #=======================================================
                                            
    # Create pseudo-observations iteratively:
    nb_pseudo_obs = 0
    
    y_new_all = []
    w_snorm = np.array(w_s) / np.sum(w_s)
    
    total_nb_obs_generated = 0
    while nb_pseudo_obs <= target_nb_pseudo_obs:
        
        #===================================================
        # Generate a batch of latent variables
        #===================================================
        
        # Draw some z^{(1)} | Theta using z^{(1)} | s, Theta
        z = np.zeros((M0, r[0]))
        
        z0_s = multivariate_normal(size = (M0, 1), \
            mean = mu_s[0].flatten(order = 'C'), cov = block_diag(*sigma_s[0]))
        z0_s = z0_s.reshape(M0, S0, r[0], order = 'C')
        
        comp_chosen = np.random.choice(S0, M0, p = w_snorm)
        for m in range(M0): # Dirty loop for the moment
            z[m] = z0_s[m, comp_chosen[m]] 
      
        #===================================================
        # Generate a batch of pseudo-observations
        #===================================================
        
        y_bin_new = []
        y_categ_new = []
        y_ord_new = []
        y_cont_new = []
        
        for mm in range(MM):
            y_bin_new.append(draw_new_bin(lambda_bin, z, nj_bin))
            y_categ_new.append(draw_new_categ(lambda_categ, z, nj_categ))
            y_ord_new.append(draw_new_ord(lambda_ord, z, nj_ord))
            y_cont_new.append(draw_new_cont(lambda_cont, z))
            
        # Stack the quantities
        y_bin_new = np.vstack(y_bin_new)
        y_categ_new = np.vstack(y_categ_new)
        y_ord_new = np.vstack(y_ord_new)
        y_cont_new = np.vstack(y_cont_new)
        
        # "Destandardize" the continous data
        y_cont_new = y_cont_new * y_std
            
        # Put them in the right order and append them to y
        type_counter = {'count': 0, 'ordinal': 0,\
                        'categorical': 0, 'continuous': 0} 
        
        y_new = np.full((M0 * MM, y.shape[1]), np.nan)
        
        # Quite dirty:
        for j, var in enumerate(var_distrib):
            if (var == 'bernoulli') or (var == 'binomial'):
                y_new[:, j] = y_bin_new[:, type_counter['count']]
                type_counter['count'] =  type_counter['count'] + 1
            elif var == 'ordinal':
                y_new[:, j] = y_ord_new[:, type_counter[var]]
                type_counter[var] =  type_counter[var] + 1
            elif var == 'categorical':
                y_new[:, j] = y_categ_new[:, type_counter[var]]
                type_counter[var] =  type_counter[var] + 1
            elif var == 'continuous':
                y_new[:, j] = y_cont_new[:, type_counter[var]]
                type_counter[var] =  type_counter[var] + 1
            else:
                raise ValueError(var, 'Type not implemented')

        #===================================================
        # Acceptation rule
        #===================================================
        
        # Check that each variable is in the good range 
        y_new_exp = np.expand_dims(y_new, 1)
        
        total_nb_obs_generated += len(y_new)
        
        mask = np.logical_and(y_new_exp >= authorized_ranges[0][np.newaxis],\
                       y_new_exp <= authorized_ranges[1][np.newaxis]) 
            
        # Keep an observation if it lies at least into one of the ranges possibility
        mask = np.any(mask.mean(2) == 1, axis = 1)   
        
        y_new = y_new[mask]
        y_new_all.append(y_new)
        nb_pseudo_obs = len(np.concatenate(y_new_all))
        
    # Keep target_nb_pseudo_obs pseudo-observations
    y_new_all = np.concatenate(y_new_all)
    y_new_all = y_new_all[:target_nb_pseudo_obs]
    
    y_all = np.vstack([y, y_new_all])
    share_kept_pseudo_obs = len(y_new_all) / total_nb_obs_generated
    
    out['y_all'] = y_all
    out['share_kept_pseudo_obs'] = share_kept_pseudo_obs
    
    return(out)

