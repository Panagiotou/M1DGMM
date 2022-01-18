# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:39:46 2021

@author: rfuchs
"""

import pandas as pd
from m1dgmm import M1DGMM
from copy import deepcopy
from oversample import error, stat_all, grad_stat, impute
# A supprimer: dist, draw_obs, pooling, stat_ord,grad_dist

from utilities import vars_contributions

from scipy.optimize import minimize   
from scipy.optimize import LinearConstraint
                        
import autograd.numpy as np

from scipy.special import logit

from sklearn.metrics.pairwise import cosine_similarity


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
    
    # !!! Hack 
    cols = y.columns
    # Formatting
    if not isinstance(nan_mask, np.ndarray): nan_mask = np.asarray(nan_mask)
    if not isinstance(y, np.ndarray): y = np.asarray(y)
    
    assert len(k) < 2 # Not implemented for deeper MDGMM for the moment
    
    # Keep complete observations
    complete_y = y[~np.isnan(y.astype(float)).any(1)]
    completed_y = deepcopy(y)
    
    out = M1DGMM(complete_y, 'auto', r, k, init, var_distrib, nj, it,\
             eps, maxstep, seed, perform_selec = perform_selec,\
                 dm = dm, max_patience = max_patience, use_silhouette = True)
        
    # Compute the associations
    vc = vars_contributions(pd.DataFrame(complete_y, columns = cols), out['Ez.y'], assoc_thr = 0.0, \
                           title = 'Contribution of the variables to the latent dimensions',\
                           storage_path = None)
        
    # Upacking the model from the M1DGMM output
    #p = y.shape[1]
    k = out['best_k']
    r = out['best_r']
    mu = out['mu'][0]
    lambda_bin = np.array(out['lambda_bin']) 
    lambda_ord = out['lambda_ord'] 
    lambda_categ = out['lambda_categ'] 
    lambda_cont = np.array(out['lambda_cont'])
    
    nj_bin = nj[pd.Series(var_distrib).isin(['bernoulli', 'binomial'])].astype(int)
    nj_ord = nj[var_distrib == 'ordinal'].astype(int)
    nj_categ = nj[var_distrib == 'categorical'].astype(int)
    
    nb_cont = np.sum(var_distrib == 'continuous')
    nb_bin =  np.sum(var_distrib == 'binomial')

    y_std = complete_y[:,var_distrib == 'continuous'].astype(float).std(axis = 0,\
                                                                    keepdims = True)
    cat_features = var_distrib != 'categorical'
    # Compute the associations between variables and use them as weights for the optimisation
    '''
    cat_features = var_distrib != 'categorical'
    assoc = compute_associations(complete_y.astype(float), nominal_columns = cat_features).values     
    np.fill_diagonal(assoc, 0.0)
    assoc = np.abs(assoc)
    weights = (assoc / assoc.sum(1, keepdims = True))
    '''
    
    assoc = cosine_similarity(vc, dense_output=True)
    np.fill_diagonal(assoc, 0.0)
    assoc = np.abs(assoc)
    weights = (assoc / assoc.sum(1, keepdims = True))

    # Keep only the max value
    #weights = np.abs(assoc)
    #thr = - np.sort(- weights, axis = 1)[:,2][..., np.newaxis]
    #weights[weights < thr] = 0
    
    #weights = np.ones((p, p))

    #==============================================
    # Optimisation sandbox
    #==============================================
    
    # Define the observation generated by the center of each cluster
    cluster_obs = [impute(mu[kk,:,0], var_distrib, lambda_bin, nj_bin, lambda_categ, nj_categ,\
                 lambda_ord, nj_ord, lambda_cont, y_std) for kk in range(k[0])]
        
    # Use only of the observed variables as references
    types = {'bin': ['bernoulli', 'binomial'], 'categ': ['categorical'],\
             'cont': ['continuous'], 'ord': 'ordinal'}

    # Gradient optimisation
    nan_indices = np.where(nan_mask.any(1))[0]
    imputed_y = np.zeros_like(y)
    numobs = y.shape[0]
    
    #************************************
    # Linear constraint to stay in the support of continuous variables
    #************************************
    
    lb = np.array([])
    ub = np.array([])
    A = np.array([[]]).reshape((0,r[0]))
    ## Corrected Binomial bounds (ub is actually +inf)

    if nb_bin > 0:
        bin_indices = var_distrib[np.logical_or(var_distrib == 'bernoulli', var_distrib == 'binomial')]
        binomial_indices = bin_indices == 'binomial'

        lb_bin = np.nanmin(y[:, var_distrib == 'binomial'], 0) 
        lb_bin = logit(lb_bin / nj_bin[binomial_indices]) - lambda_bin[binomial_indices,0]
        ub_bin = np.nanmax(y[:, var_distrib == 'binomial'], 0)
        ub_bin = logit(ub_bin / nj_bin[binomial_indices]) - lambda_bin[binomial_indices,0]
        A_bin = lambda_bin[binomial_indices,1:]

        ## Concatenate the constraints
        lb = np.concatenate([lb, lb_bin])
        ub = np.concatenate([ub, ub_bin])
        A = np.concatenate([A, A_bin], axis = 0)

    if nb_cont > 0:
        ## Corrected Gaussian bounds (ub is actually +inf)
        lb_cont = np.nanmin(y[:, var_distrib == 'continuous'], 0) / y_std[0] - lambda_cont[:,0]
        ub_cont = np.nanmax(y[:, var_distrib == 'continuous'], 0) / y_std[0] - lambda_cont[:,0]
        A_cont = lambda_cont[:,1:]
        
        ## Concatenate the constraints
        lb = np.concatenate([lb, lb_cont])
        ub = np.concatenate([ub, ub_cont])
        A = np.concatenate([A, A_cont], axis = 0)
        
    ## Concatenate the constraints
    #lb = np.concatenate([lb_bin, lb_cont])
    #ub = np.concatenate([ub_bin, ub_cont])
    #A = np.concatenate([A_bin, A_cont], axis = 0)

    lc = LinearConstraint(A, lb, ub, keep_feasible = True)
    
    zz = []
    fun = []
    for i in range(numobs):
        if i in nan_indices: 
            
            # Design the nan masks for the optimisation process
            nan_mask_i = nan_mask[i]
            weights_i = weights[nan_mask_i].mean(0)

            # Look for the best starting point
            cluster_dist = [error(y[i, ~nan_mask_i], obs[~nan_mask_i],\
                            cat_features[~nan_mask_i], weights_i)\
                            for obs in cluster_obs]
            z02 = mu[np.argmin(cluster_dist),:,0]
            
            # Formatting
            vars_i = {type_alias: np.where(~nan_mask_i[np.isin(var_distrib, vartype)])[0] \
                             for type_alias, vartype in types.items()}
              
            complete_categ = [l for idx, l in enumerate(lambda_categ) if idx in vars_i['categ']]
            complete_ord = [l for idx, l in enumerate(lambda_ord) if idx in vars_i['ord']]
            
            
            # Find the most promising regions
            '''
            opt = minimize(stat_all, z02, \
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
            '''
                
            opt = minimize(stat_all, z02, \
                   args = (y[i, ~nan_mask_i], var_distrib[~nan_mask_i],\
                   weights_i[~nan_mask_i],\
                   lambda_bin[vars_i['bin']], nj_bin[vars_i['bin']],\
                   complete_categ,\
                   nj_categ[vars_i['categ']],\
                   complete_ord,\
                   nj_ord[vars_i['ord']],\
                   lambda_cont[vars_i['cont']], y_std[:, vars_i['cont']]), 
                   tol = eps, method='trust-constr', jac = grad_stat,\
                   constraints = lc, 
                   options = {'maxiter': 1000})
                
            z = opt.x
            zz.append(z)
            fun.append(opt.fun)
                            
            imputed_y[i] = impute(z, var_distrib, lambda_bin, nj_bin, lambda_categ, nj_categ,\
                         lambda_ord, nj_ord, lambda_cont, y_std)
                
        else:
            imputed_y[i] = y[i] 

        
    completed_y = np.where(nan_mask, imputed_y, y)
  
    out['completed_y'] = completed_y
    out['zz'] =  zz
    out['fun'] = fun
    return(out)

