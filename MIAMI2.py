# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:11:28 2021

@author: RobF
"""

            

  
import pandas as pd
from m1dgmm import M1DGMM
from oversample import draw_new_bin, draw_new_ord,\
                            draw_new_categ, draw_new_cont
                            
import autograd.numpy as np



def MIAMI(y, n_clusters, r, k, init, var_distrib, nj, authorized_ranges,\
          target_nb_pseudo_obs = 500, it = 50, \
          eps = 1E-05, maxstep = 100, seed = None, perform_selec = True,\
              dm = None, max_patience = 1): # dm: Hack to remove
    
    ''' Fit a Generalized Linear Mixture of Latent Variables Model (GLMLVM)
    
    y (numobs x p ndarray): The observations containing mixed variables
    n_clusters (int): The number of clusters to look for in the data
    r (list): The dimension of latent variables through the first 2 layers
    k (list): The number of components of the latent Gaussian mixture layers
    init (dict): The initialisation parameters for the algorithm
    var_distrib (p 1darray): An array containing the types of the variables in y 
    nj (p 1darray): For binary/count data: The maximum values that the variable can take. 
                    For ordinal data: the number of different existing categories for each variable
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
    best_z = out['best_z']
    k = out['best_k']
    r = out['best_r']
    w_s = out['best_w_s'] 
    lambda_bin = out['lambda_bin'] 
    lambda_ord = out['lambda_ord'] 
    lambda_categ = out['lambda_categ'] 
    lambda_cont = out['lambda_cont'] 
    
    nj_bin = nj[pd.Series(var_distrib).isin(['bernoulli', 'binomial'])].astype(int)
    nj_ord = nj[var_distrib == 'ordinal'].astype(int)
    nj_categ = nj[var_distrib == 'categorical'].astype(int)

    y_std = y[:,var_distrib == 'continuous'].std(axis = 0, keepdims = True)
    
    M0 = best_z.shape[0]
    S0 = best_z.shape[-1]
        
    #=======================================================
    # Data augmentation part
    #=======================================================
                                        
    # Best z_s might be of a wrong dimension after architecture selection:
    # Have to deal with it in the future.
    
    # Create pseudo-observations iteratively:
    nb_pseudo_obs = 0
    
    y_new_all = []
    w_snorm = np.array(w_s) / np.sum(w_s)
    
    total_nb_obs_generated = 0
    while nb_pseudo_obs <= target_nb_pseudo_obs:
        
        #===================================================
        # Generate a batch of pseudo-observations
        #===================================================
        
        # Draw some z^{(1)} | Theta
        comp_chosen = np.random.choice(S0, M0, p = w_snorm)
        z = np.zeros((M0, S0))
        for m in range(M0): # Dirty loop for the moment
            z = best_z[:,:,comp_chosen[m]]
        
        # Draw the new y
        y_bin_new = draw_new_bin(lambda_bin, z, nj_bin)
        y_categ_new = draw_new_categ(lambda_categ, z, nj_categ)
        y_ord_new = draw_new_ord(lambda_ord, z, nj_ord)
        y_cont_new = draw_new_cont(lambda_cont, z)
        
        # "Destandardize" the continous data
        y_cont_new = y_cont_new * y_std
            
        # Put them in the right order and append them to y
        type_counter = {'count': 0, 'ordinal': 0,\
                        'categorical': 0, 'continuous': 0} 
        
        y_new = np.full((z.shape[0], y.shape[1]), np.nan)
        
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
        #if np.sum(mask) !=0:
            #print('Add', np.sum(mask))
        
        y_new = y_new[mask]
        y_new_all.append(y_new)
        nb_pseudo_obs = len(np.concatenate(y_new_all))
        
        
    y_new_all = np.concatenate(y_new_all)
    
    y_all = np.vstack([y, y_new_all])
    share_kept_pseudo_obs = len(y_new_all) / total_nb_obs_generated
    
    out['y_all'] = y_all
    out['share_kept_pseudo_obs'] = share_kept_pseudo_obs
    
    return(out)

