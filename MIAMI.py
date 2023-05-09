# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 08:57:09 2022

@author: rfuchs
"""

import pandas as pd
from m1dgmm import M1DGMM
from oversample import draw_new_bin, draw_new_ord,\
                       draw_new_categ, draw_new_cont,\
                       impute, fz, generate_random
        
from MCEM_DGMM import draw_z_s                        
from utilities import vars_contributions

from scipy.special import logit
#from shapely.geometry import Polygon
from oversample import solve_convex_set                 
import autograd.numpy as np

from scipy.spatial.qhull import QhullError

from autograd.numpy.random import multivariate_normal
from scipy.linalg import block_diag

from copy import deepcopy

def MIAMI(y, n_clusters, r, k, init, var_distrib, nj, authorized_ranges,\
          target_nb_pseudo_obs = 500, nb_points=1, it = 50, \
          eps = 1E-05, maxstep = 100, seed = None, perform_selec = True,\
              dm = [], max_patience = 1, pretrained_model = False): # dm, pretrained_model: Hack to remove
    
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
    if not isinstance(y, np.ndarray): y = np.asarray(y)
    
    assert len(k) < 2 # Not implemented for deeper MDGMM for the moment
    
    
    if pretrained_model:
        # !!! TO DO: Delete the useless keys
        out = deepcopy(init)
    else:
        out = M1DGMM(y, n_clusters, r, k, init, var_distrib, nj, it,\
             eps, maxstep, seed, perform_selec = perform_selec,\
                 dm = dm, max_patience = max_patience, use_silhouette = True)
    
    # Compute the associations
    #vars_contributions(pd.DataFrame(y, columns = cols), out['Ez.y'], assoc_thr = 0.0, \
                           #title = 'Contribution of the variables to the latent dimensions',\
                           #storage_path = None)
    
        
    # Upacking the model from the M1DGMM output
    p = y.shape[1]
    k = out['best_k']
    r = out['best_r']
    mu = out['mu'][0]
    sigma = out['sigma'][0]
    w = out['best_w_s']
    #eta = out['eta'][0]

    #Ez_y = out['Ez.y']
    
    lambda_bin = np.array(out['lambda_bin']) 
    lambda_ord = out['lambda_ord'] 
    lambda_categ = out['lambda_categ'] 
    lambda_cont = np.array(out['lambda_cont'])
    
    nj_bin = nj[pd.Series(var_distrib).isin(['bernoulli', 'binomial'])].astype(int)
    nj_ord = nj[var_distrib == 'ordinal'].astype(int)
    nj_categ = nj[var_distrib == 'categorical'].astype(int)

    y_std = y[:,var_distrib == 'continuous'].astype(float).std(axis = 0,\
                                                                    keepdims = True)
    
    # nb_points = 200
    #=======================================================
    # Data augmentation part
    #=======================================================
                                            
    # Create pseudo-observations iteratively:
    nb_pseudo_obs = 0
    
    y_new_all = []
    zz = []
    
    total_nb_obs_generated = 0
    while nb_pseudo_obs <= target_nb_pseudo_obs:
        
        #===================================================
        # Generate a batch of latent variables (try)
        #===================================================
        
        '''
        # Simulate points in the Polynom
        pts = generate_random(nb_points, polygon)
        pts = np.array([np.array([p.x, p.y]) for p in pts])
        
        # Compute their density and resample them
        pts_density = fz(pts, mu, sigma, w)
        pts_density = pts_density / pts_density.sum(keepdims = True) # Normalized the pdfs
        
        idx = np.random.choice(np.arange(nb_points), size = target_nb_pseudo_obs,\
                               p = pts_density, replace=True)
        z = pts[idx]
        '''
        #===================================================
        # Generate a batch of latent variables
        #===================================================
        
        # Draw some z^{(1)} | Theta using z^{(1)} | s, Theta
        z = np.zeros((nb_points, r[0]))

        z0_s = multivariate_normal(size = (nb_points, 1), \
            mean = mu.flatten(order = 'C'), cov = block_diag(*sigma))
        z0_s = z0_s.reshape(nb_points, k[0], r[0], order = 'C')

        comp_chosen = np.random.choice(k[0], nb_points, p = w / w.sum())
        for m in range(nb_points): # Dirty loop for the moment
            z[m] = z0_s[m, comp_chosen[m]] 

        #===================================================
        # Draw pseudo-observations
        #===================================================
                
        y_bin_new = []
        y_categ_new = []
        y_ord_new = []
        y_cont_new = []
        

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
        
        y_new = np.full((nb_points, y.shape[1]), np.nan)
        
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
        if not authorized_ranges is None:
            # Check that each variable is in the good range 
            y_new_exp = np.expand_dims(y_new, 1)
            
            
            mask = np.logical_and(y_new_exp >= authorized_ranges[0][np.newaxis],\
                        y_new_exp <= authorized_ranges[1][np.newaxis]) 
                
            # Keep an observation if it lies at least into one of the ranges possibility
            mask = np.any(mask.mean(2) == 1, axis = 1)   
            
            y_new = y_new[mask]

        y_new_all.append(y_new)

        total_nb_obs_generated += len(y_new)

        nb_pseudo_obs = len(np.concatenate(y_new_all))

        if not authorized_ranges is None:
            zz.append(z[mask])
        else:
            zz.append(z)
        #print(nb_pseudo_obs)
        
    # Keep target_nb_pseudo_obs pseudo-observations
    y_new_all = np.concatenate(y_new_all)
    y_new_all = y_new_all[:target_nb_pseudo_obs]
    
    #y_all = np.vstack([y, y_new_all])
    share_kept_pseudo_obs = len(y_new_all) / total_nb_obs_generated
    
    out['zz'] = zz
    out['y_all'] = y_new_all
    out['share_kept_pseudo_obs'] = share_kept_pseudo_obs

    return(out)



    '''
    y_new = [impute(zz, var_distrib, lambda_bin, nj_bin, lambda_categ, nj_categ,\
                 lambda_ord, nj_ord, lambda_cont, y_std)[is_constrained] for zz in z]
        
  
    import matplotlib.pyplot as plt
    plt.plot(*polygon.exterior.xy)
    plt.scatter(pts[:,0], pts[:,1], color = 'orange')
    plt.scatter(z[:,0], z[:,1], color = 'green')
    '''