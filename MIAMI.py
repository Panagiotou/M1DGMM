# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:11:28 2021

@author: RobF
"""

from copy import deepcopy

from numeric_stability import ensure_psd
from parameter_selection import r_select, k_select 

from identifiability_DGMM import identifiable_estim_DGMM, compute_z_moments, \
    diagonal_cond
                         
from MCEM_DGMM import draw_z_s, fz2_z1s, draw_z2_z1s, fz_ys,\
    E_step_DGMM, M_step_DGMM

from MCEM_GLLVM import draw_zl1_ys, fy_zl1, E_step_GLLVM, \
        bin_params_GLLVM, ord_params_GLLVM, categ_params_GLLVM,\
            cont_params_GLLVM
            
from oversample import draw_new_bin, draw_new_ord,\
                            draw_new_categ, draw_new_cont
  
from hyperparameters_selection import M_growth, look_for_simpler_network
from utilities import compute_path_params, compute_chsi, compute_rho

import autograd.numpy as np
from autograd.numpy import transpose as t
from autograd.numpy import newaxis as n_axis

from gower import gower_matrix
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


def miami(y, n_clusters, r, k, init, var_distrib, nj, authorized_ranges,\
          target_nb_pseudo_obs = 500, it = 50, \
          eps = 1E-05, maxstep = 100, seed = None, perform_selec = True): 
    
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
    ------------------------------------------------------------------------------------------------
    returns (dict): The predicted classes, the likelihood through the EM steps
                    and a continuous representation of the data
    '''

    prev_lik = - 1E16
    best_lik = -1E16
    
    best_sil = -1 
    new_sil = -1 
    
    best_k = deepcopy(k)
    best_r = deepcopy(r)
    
    tol = 0.01
    max_patience = 1
    patience = 0
    
    # Initialize the parameters
    eta = deepcopy(init['eta'])
    psi = deepcopy(init['psi'])
    lambda_bin = deepcopy(init['lambda_bin'])
    lambda_ord = deepcopy(init['lambda_ord'])
    lambda_cont = deepcopy(init['lambda_cont'])
    lambda_categ = deepcopy(init['lambda_categ'])

    H = deepcopy(init['H'])
    w_s = deepcopy(init['w_s']) # Probability of path s' through the network for all s' in Omega
   
    numobs = len(y)
    likelihood = []
    it_num = 0
    ratio = 1000
    np.random.seed = seed
        
    # Dispatch variables between categories
    y_bin = y[:, np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nj_bin = nj[np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')].astype(int)
    nb_bin = len(nj_bin)
        
    y_ord = y[:, var_distrib == 'ordinal']    
    nj_ord = nj[var_distrib == 'ordinal'].astype(int)
    nb_ord = len(nj_ord)
    
    y_categ = y[:, var_distrib == 'categorical']
    nj_categ = nj[var_distrib == 'categorical'].astype(int)
    nb_categ = len(nj_categ)    
    
    y_cont = y[:, var_distrib == 'continuous'] 
    nb_cont = y_cont.shape[1]
    
    # Set y_count standard error to 1
    y_std = y_cont.std(axis = 0, keepdims = True)
    y_cont = y_cont / y_std
    
    L = len(k)
    k_aug = k + [1]
    S = np.array([np.prod(k_aug[l:]) for l in range(L + 1)])    
    M = M_growth(1, r, numobs)
   
    assert nb_bin + nb_ord + nb_cont + nb_categ > 0 
    if nb_bin + nb_ord + nb_cont + nb_categ != len(var_distrib):
        raise ValueError('Some variable types were not understood,\
                         existing types are: continuous, categorical,\
                         ordinal, binomial and bernoulli')

    # Compute the Gower matrix
    cat_features = np.logical_or(var_distrib == 'categorical', var_distrib == 'bernoulli')
    dm = gower_matrix(y, cat_features = cat_features)
                
    while (it_num < it) & ((ratio > eps) | (patience <= max_patience)):
        print(it_num)

        # The clustering layer is the one used to perform the clustering 
        # i.e. the layer l such that k[l] == n_clusters
        clustering_layer = np.argmax(np.array(k) == n_clusters)

        #####################################################################################
        ################################# S step ############################################
        #####################################################################################

        #=====================================================================
        # Draw from f(z^{l} | s, Theta) for all s in Omega
        #=====================================================================  
        
        mu_s, sigma_s = compute_path_params(eta, H, psi)
        sigma_s = ensure_psd(sigma_s)
        z_s, zc_s = draw_z_s(mu_s, sigma_s, eta, M)
         
        #========================================================================
        # Draw from f(z^{l+1} | z^{l}, s, Theta) for l >= 1
        #========================================================================
        
        chsi = compute_chsi(H, psi, mu_s, sigma_s)
        chsi = ensure_psd(chsi)
        rho = compute_rho(eta, H, psi, mu_s, sigma_s, zc_s, chsi)

        # In the following z2 and z1 will denote z^{l+1} and z^{l} respectively
        z2_z1s = draw_z2_z1s(chsi, rho, M, r)
                   
        #=======================================================================
        # Compute the p(y| z1) for all variable categories
        #=======================================================================
        
        py_zl1 = fy_zl1(lambda_bin, y_bin, nj_bin, lambda_ord, y_ord, nj_ord, \
                        lambda_categ, y_categ, nj_categ, y_cont, lambda_cont, z_s[0])
        
        #========================================================================
        # Draw from p(z1 | y, s) proportional to p(y | z1) * p(z1 | s) for all s
        #========================================================================
                
        zl1_ys = draw_zl1_ys(z_s, py_zl1, M)
                
        #####################################################################################
        ################################# E step ############################################
        #####################################################################################
        
        #=====================================================================
        # Compute conditional probabilities used in the appendix of asta paper
        #=====================================================================
        
        pzl1_ys, ps_y, p_y = E_step_GLLVM(z_s[0], mu_s[0], sigma_s[0], w_s, py_zl1)

        #=====================================================================
        # Compute p(z^{(l)}| s, y). Equation (5) of the paper
        #=====================================================================
        
        pz2_z1s = fz2_z1s(t(pzl1_ys, (1, 0, 2)), z2_z1s, chsi, rho, S)
        pz_ys = fz_ys(t(pzl1_ys, (1, 0, 2)), pz2_z1s)
                
        
        #=====================================================================
        # Compute MFA expectations
        #=====================================================================
        
        Ez_ys, E_z1z2T_ys, E_z2z2T_ys, EeeT_ys = \
            E_step_DGMM(zl1_ys, H, z_s, zc_s, z2_z1s, pz_ys, pz2_z1s, S)


        ###########################################################################
        ############################ M step #######################################
        ###########################################################################
             
        #=======================================================
        # Compute MFA Parameters 
        #=======================================================

        w_s = np.mean(ps_y, axis = 0)      
        eta, H, psi = M_step_DGMM(Ez_ys, E_z1z2T_ys, E_z2z2T_ys, EeeT_ys, ps_y, H, k)

        #=======================================================
        # Identifiability conditions
        #======================================================= 

        # Update eta, H and Psi values
        H = diagonal_cond(H, psi)
        Ez, AT = compute_z_moments(w_s, eta, H, psi)
        eta, H, psi = identifiable_estim_DGMM(eta, H, psi, Ez, AT)
        
        del(Ez)
        
        #=======================================================
        # Compute GLLVM Parameters
        #=======================================================
                        
        lambda_bin = bin_params_GLLVM(y_bin, nj_bin, lambda_bin, ps_y, pzl1_ys, z_s[0], AT[0],\
                     tol = tol, maxstep = maxstep)
                 
        lambda_ord = ord_params_GLLVM(y_ord, nj_ord, lambda_ord, ps_y, pzl1_ys, z_s[0], AT[0],\
                     tol = tol, maxstep = maxstep)
            
        lambda_categ = categ_params_GLLVM(y_categ, nj_categ, lambda_categ, ps_y, pzl1_ys, z_s[0], AT[0],\
                     tol = tol, maxstep = maxstep)

        lambda_cont = cont_params_GLLVM(y_cont, lambda_cont, ps_y, pzl1_ys, z_s[0], AT[0],\
                     tol = tol, maxstep = maxstep)

        ###########################################################################
        ################## Clustering parameters updating #########################
        ###########################################################################
          
        new_lik = np.sum(np.log(p_y))
        likelihood.append(new_lik)
        ratio = (new_lik - prev_lik)/abs(prev_lik)
        print(likelihood)
        
        idx_to_sum = tuple(set(range(1, L + 1)) - set([clustering_layer + 1]))
        psl_y = ps_y.reshape(numobs, *k, order = 'C').sum(idx_to_sum) 

        temp_class = np.argmax(psl_y, axis = 1)
        try:
            new_sil = silhouette_score(dm, temp_class, metric = 'precomputed')
        except ValueError:
            new_sil = -1
            
        if best_sil < new_sil:
            Ez_y = (ps_y[..., n_axis] * Ez_ys[clustering_layer]).sum(1)
            best_sil = deepcopy(new_sil)
            classes = deepcopy(temp_class)
            best_z = deepcopy(z_s[0])

            #plt.figure(figsize=(8,8))
            #plt.scatter(Ez_y[:, 0], Ez_y[:, 1], c = classes)
            #plt.show()

        # Refresh the classes only if they provide a better explanation of the data
        if best_lik < new_lik:
            best_lik = deepcopy(prev_lik)
                               
        if prev_lik < new_lik:
            patience = 0
            M = M_growth(it_num + 2, r, numobs)
        else:
            patience += 1
                          
        ###########################################################################
        ######################## Parameter selection  #############################
        ###########################################################################
        
        is_not_min_specif = not(np.all(np.array(k) == n_clusters) & np.array_equal(r, [2,1]))
        
        if look_for_simpler_network(it_num) & perform_selec & is_not_min_specif:
            r_to_keep = r_select(y_bin, y_ord, y_categ, y_cont, zl1_ys, z2_z1s, w_s)
            
            # If r_l == 0, delete the last l + 1: layers
            new_L = np.sum([len(rl) != 0 for rl in r_to_keep]) - 1 
            
            k_to_keep = k_select(w_s, k, new_L, clustering_layer)
    
            is_L_unchanged = (L == new_L)
            is_r_unchanged = np.all([len(r_to_keep[l]) == r[l] for l in range(new_L + 1)])
            is_k_unchanged = np.all([len(k_to_keep[l]) == k[l] for l in range(new_L)])
              
            is_selection = not(is_r_unchanged & is_k_unchanged & is_L_unchanged)
            
            assert new_L > 0
            
            if is_selection:           
                
                eta = [eta[l][k_to_keep[l]] for l in range(new_L)]
                eta = [eta[l][:, r_to_keep[l]] for l in range(new_L)]
                
                H = [H[l][k_to_keep[l]] for l in range(new_L)]
                H = [H[l][:, r_to_keep[l]] for l in range(new_L)]
                H = [H[l][:, :, r_to_keep[l + 1]] for l in range(new_L)]
                
                psi = [psi[l][k_to_keep[l]] for l in range(new_L)]
                psi = [psi[l][:, r_to_keep[l]] for l in range(new_L)]
                psi = [psi[l][:, :, r_to_keep[l]] for l in range(new_L)]
                
                if nb_bin > 0:
                    # Add the intercept:
                    bin_r_to_keep = np.concatenate([[0], np.array(r_to_keep[0]) + 1]) 
                    lambda_bin = lambda_bin[:, bin_r_to_keep]
                 
                if nb_ord > 0:
                    # Intercept coefficients handling is a little more complicated here
                    lambda_ord_intercept = [lambda_ord_j[:-r[0]] for lambda_ord_j in lambda_ord]
                    Lambda_ord_var = np.stack([lambda_ord_j[-r[0]:] for lambda_ord_j in lambda_ord])
                    Lambda_ord_var = Lambda_ord_var[:, r_to_keep[0]]
                    lambda_ord = [np.concatenate([lambda_ord_intercept[j], Lambda_ord_var[j]])\
                                  for j in range(nb_ord)]
    
                # To recheck
                if nb_cont > 0:
                    # Add the intercept:
                    cont_r_to_keep = np.concatenate([[0], np.array(r_to_keep[0]) + 1]) 
                    lambda_cont = lambda_cont[:, cont_r_to_keep]  
                    
                if nb_categ > 0:
                    lambda_categ_intercept = [lambda_categ[j][:, 0]  for j in range(nb_categ)]
                    Lambda_categ_var = [lambda_categ_j[:,-r[0]:] for lambda_categ_j in lambda_categ]
                    Lambda_categ_var = [lambda_categ_j[:, r_to_keep[0]] for lambda_categ_j in lambda_categ]

                    lambda_categ = [np.hstack([lambda_categ_intercept[j][..., n_axis], Lambda_categ_var[j]])\
                                   for j in range(nb_categ)]  

                w = w_s.reshape(*k, order = 'C')
                new_k_idx_grid = np.ix_(*k_to_keep[:new_L])
                
                # If layer deletion, sum the last components of the paths
                if L > new_L: 
                    deleted_dims = tuple(range(L)[new_L:])
                    w_s = w[new_k_idx_grid].sum(deleted_dims).flatten(order = 'C')
                else:
                    w_s = w[new_k_idx_grid].flatten(order = 'C')
    
                w_s /= w_s.sum()
    
                k = [len(k_to_keep[l]) for l in range(new_L)]
                r = [len(r_to_keep[l]) for l in range(new_L + 1)]
                
                k_aug = k + [1]
                S = np.array([np.prod(k_aug[l:]) for l in range(new_L + 1)])    
                L = new_L

                best_r = deepcopy(r)
                best_k = deepcopy(k)
                
                patience = 0
                
                # Identifiability conditions
                H = diagonal_cond(H, psi)
                Ez, AT = compute_z_moments(w_s, eta, H, psi)
                eta, H, psi = identifiable_estim_DGMM(eta, H, psi, Ez, AT)
        
                del(Ez)
                         
            print('New architecture:')
            print('k', k)
            print('r', r)
            print('L', L)
            print('S',S)
            print("w_s", len(w_s))
            
        prev_lik = deepcopy(new_lik)
        it_num = it_num + 1
        
        
    #=======================================================
    # Data augmentation part
    #=======================================================
                                        
    # Best z_s might be of a wrong dimension after architecture selection:
    # Have to deal with it in the future.
    
    # Create pseudo-observations iteratively:
    nb_pseudo_obs = 0
    
    y_new_all = []
    
    while nb_pseudo_obs <= target_nb_pseudo_obs:
        
        #===================================================
        # Generate a batch of pseudo-observations
        #===================================================
        
        # Draw some z^{(1)} | Theta
        comp_chosen = np.random.choice(S[0], M[0], p=w_s)
        z = np.zeros((M[0], S[0]))
        for m in range(M[0]): # Dirty loop for the moment
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
        
        mask = np.logical_and(y_new_exp >= authorized_ranges[0][np.newaxis],\
                       y_new_exp <= authorized_ranges[1][np.newaxis]) 
            
        # Keep an observation if it lies at least into one of the ranges possibility
        mask = np.any(mask.mean(2) == 1, axis = 1)    
        
        y_new = y_new[mask]
        y_new_all.append(y_new)
        nb_pseudo_obs = len(np.concatenate(y_new_all))
        
    y_new_all = np.concatenate(y_new_all)
    
    y_all = np.vstack([y, y_new_all])
    
    out = dict(likelihood = likelihood, y = y_all, classes = classes, Ez_y = Ez_y, \
               best_r = best_r, best_k = best_k)
    return(out)

