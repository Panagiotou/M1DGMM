# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:39:46 2021

@author: rfuchs
"""

import matplotlib.pyplot as plt 


import pandas as pd
from m1dgmm import M1DGMM
from copy import deepcopy
from oversample import dist, error, draw_obs, pooling, stat_all, stat_ord, grad_stat, impute#,grad_dist
from scipy.linalg import block_diag
from oversample import draw_new_bin, draw_new_ord,\
                            draw_new_categ, draw_new_cont

from gower import gower_matrix
from scipy.stats import mode 
from scipy.optimize import minimize                           
import autograd.numpy as np
from autograd.numpy.random import multivariate_normal

from dython.nominal import associations, compute_associations


def MI2AMI(y, n_clusters, r, k, init, var_distrib, nj,\
          nan_mask, target_nb_pseudo_obs = 500, it = 50, \
          eps = 1E-05, maxstep = 100, seed = None, perform_selec = True,\
          dm = [], max_patience = 1): # dm: Hack to remove
    
    ''' Complete the missing values using a trained M1DGMM
    
    y (numobs x p ndarray): The observations containing mixed variables
    n_clusters (int): The number of clusters to look for in the data
    r (list): The dimension of latent variables through the first 2 layers
    k (list): The number of components of the latent Gaussian mixture layers
    init (dict): The initialisation parameters for the algorithm
    var_distrib (p 1darray): An array containing the types of the variables in y 
    nj (p 1darray): For binary/count data: The maximum values that the variable can take. 
                    For ordinal data: the number of different existing categories for each variable
    nan_mask (ndarray): A mask array equal to True when the observation value is missing False otherwise
    target_nb_pseudo_obs (int): The number of pseudo-observations to generate         
    it (int): The maximum number of MCEM iterations of the algorithm
    eps (float): If the likelihood increase by less than eps then the algorithm stops
    maxstep (int): The maximum number of optimisation step for each variable
    seed (int): The random state seed to set (Only for numpy generated data for the moment)
    perform_selec (Bool): Whether to perform architecture selection or not
    dm (np array): The distance matrix of the observations. If not given M1DGMM computes it
    n_neighbors (int): The number of neighbors to use for NA imputation
    ------------------------------------------------------------------------------------------------
    returns (dict): The predicted classes, the likelihood through the EM steps
                    and a continuous representation of the data
    '''

    # Formatting
    if not isinstance(nan_mask, np.ndarray): nan_mask = np.asarray(nan_mask)
    if not isinstance(y, np.ndarray): y = np.asarray(y)
    
    assert len(k) < 2 # Not implemented for deeper MDGMM for the moment
    
    # Keep complete observations
    complete_y = y[~np.isnan(y.astype(float)).any(1)]
    completed_y = deepcopy(y)
    
    out = M1DGMM(complete_y, 'auto', r, k, init, var_distrib, nj, it,\
             eps, maxstep, seed, perform_selec = perform_selec,\
                 dm = dm, max_patience = max_patience, use_silhouette = False)
        
    # Upacking the model from the M1DGMM output
    k = out['best_k']
    r = out['best_r']
    lambda_bin = out['lambda_bin'] 
    lambda_ord = out['lambda_ord'] 
    lambda_categ = out['lambda_categ'] 
    lambda_cont = out['lambda_cont'] 
    
    nj_bin = nj[pd.Series(var_distrib).isin(['bernoulli', 'binomial'])].astype(int)
    nj_ord = nj[var_distrib == 'ordinal'].astype(int)
    nj_categ = nj[var_distrib == 'categorical'].astype(int)

    y_std = complete_y[:,var_distrib == 'continuous'].astype(float).std(axis = 0,\
                                                                    keepdims = True)
    
    # Compute the associations between variables and use them as weights for the optimisation
    cat_features = var_distrib != 'categorical'
    assoc = compute_associations(complete_y.astype(float), nominal_columns = cat_features).values     
    np.fill_diagonal(assoc, 0.0)
    assoc = np.abs(assoc)
    weights = (assoc / assoc.sum(1, keepdims = True))


    #==============================================
    # Optimisation sandbox
    #==============================================

    z0 = np.full(r[0], 0)
    # Use only of the observed variables as references
    types = {'bin': ['bernoulli', 'binomial'], 'categ': ['categorical'],\
             'cont': ['continuous'], 'ord': 'ordinal'}

    # Gradient optimisation
    nan_indices = np.where(nan_mask.any(1))[0]
    imputed_y = np.zeros_like(y)
    numobs = y.shape[0]
    
    for i in range(numobs):
        if i in nan_indices: 
            
            # Design the nan masks for the optimisation process
            nan_mask_i = nan_mask[i]

            vars_i = {type_alias: np.where(~nan_mask_i[np.isin(var_distrib, vartype)])[0] \
                             for type_alias, vartype in types.items()}
              
            complete_categ = [l for idx, l in enumerate(lambda_categ) if idx in vars_i['categ']]
            complete_ord = [l for idx, l in enumerate(lambda_ord) if idx in vars_i['ord']]
            
            weights_i = weights[nan_mask_i].mean(0)

            # Find the most promising regions
            opt2 = minimize(stat_all, z0[np.newaxis], \
                   args = (y[i, ~nan_mask_i], var_distrib[~nan_mask_i],\
                   weights_i[~nan_mask_i],\
                   lambda_bin[vars_i['bin']], nj_bin[vars_i['bin']],\
                   complete_categ,\
                   nj_categ[vars_i['categ']],\
                   complete_ord,\
                   nj_ord[vars_i['ord']],\
                   lambda_cont[vars_i['cont']], y_std[:, vars_i['cont']]), 
                   tol = eps, method='BFGS', jac = grad_stat,\
                   options = {'maxiter': 1000})
                
            z = opt2.x
                            
            imputed_y[i] = impute(z, var_distrib, lambda_bin, nj_bin, lambda_categ, nj_categ,\
                         lambda_ord, nj_ord, lambda_cont, y_std)
                
        else:
            imputed_y[i] = y[i] 

        
    completed_y = np.where(nan_mask, imputed_y, y)
    
    '''    
    while nb_pseudo_obs < target_nb_pseudo_obs:
        
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

        y_new_all.append(y_new)
        nb_pseudo_obs = len(np.concatenate(y_new_all))
       
        
    #========================================================
    # Inferring missing data using the neighrest neighbours
    #========================================================
    
    # Keep target_nb_pseudo_obs pseudo-observations
    y_new_all = np.concatenate(y_new_all)
    y_new_all = y_new_all[:target_nb_pseudo_obs]
        
    # Can compute the average distance has a quality criterion
    numobs = len(completed_y)
    for i in range(numobs):
        if nan_mask[i].any():
            missing_vars = np.where(nan_mask[i])[0]
            
            # Compute the k-closest observations            
            cat_features = np.logical_or(var_distrib[~nan_mask[i]] == 'categorical',\
                                       var_distrib[~nan_mask[i]] == 'bernoulli')
            
               
            dist = gower_matrix(y[i][~nan_mask[i]][np.newaxis], y_new_all[:,~nan_mask[i]] ,\
                         cat_features = cat_features)[0]
                
            idx = np.argpartition(dist, n_neighbors)
            neighbors = y_new_all[idx[:n_neighbors]]
            
            for j in missing_vars:
                if var_distrib[j] == 'continuous':
                    completed_y[i,j] = neighbors[:,j].mean()
                elif var_distrib[j] == 'categorical':
                    #print('categ')
                    #print(mode(neighbors[:,j]))
                    #print('------------')
                    completed_y[i,j] = str(int(mode(neighbors[:,j])[0][0]))
                else:
                    completed_y[i,j] = neighbors[:,j].mean().round(0)
    '''
    out['completed_y'] = completed_y
    return(out)

