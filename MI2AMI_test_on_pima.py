# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:45:07 2022

@author: rfuchs
"""

import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/M1DGMM')

import pandas as pd
from gower import gower_matrix
import matplotlib .pyplot as plt

from mi2ami import MI2AMI
from init_params import dim_reduce_init
from utilities import vars_contributions, obs_representation
from data_preprocessing import compute_nj, data_processing
import autograd.numpy as np

res = 'C:/Users/rfuchs/Documents/These/Stats/MIAMI/Missing_data/MI2AMI/Diabetes/'

###############################################################################
###############     PIMA    vizualisation      #######################
###############################################################################

dtypes_dict = {'continuous': float, 'categorical': str, 'ordinal': float,\
              'bernoulli': int, 'binomial': int}

###############################################################################
######################## Pima data vizualisation    #########################
###############################################################################

#===========================================#
# Importing data
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/MIAMI/Missing_data/Data/Diabetes')

for dataset_name in ['MCAR30']:
#dataset_name = 'MCAR30'
    
    full_pima = pd.read_csv('Diabetes.csv', sep = ',')
    p = full_pima.shape[1]
    y = pd.read_csv(dataset_name + '.csv', sep = ';', decimal = ',').iloc[:,1:].astype(float)
    
    var_distrib = np.array(['binomial', 'continuous', 'continuous', 'continuous',\
                            'continuous', 'continuous', 'continuous', 'continuous',\
                            'bernoulli']) 
     
    #===========================================#
    # Formating the data
    #===========================================#
    
    nan_mask = y.isnull()
    cat_features = var_distrib == 'categorical'
    y, le_dict  = data_processing(y, var_distrib)
    y = y.where(~nan_mask, np.nan)
              
    
    nj, nj_bin, nj_ord, nj_categ = compute_nj(full_pima, var_distrib)
    nb_cont = np.sum(var_distrib == 'continuous')
    
    # Feature category (cf)
    dtype = {y.columns[j]: dtypes_dict[var_distrib[j]] for j in range(p)}
    
    full_pima = full_pima.astype(dtype, copy=True)
    complete_y = y[~y.isna().any(1)].astype(dtype, copy=True)
    
    # Defining distance matrix
    dm = gower_matrix(complete_y, cat_features = cat_features) 
    
    #===========================================#
    # Hyperparameters
    #===========================================# 
    
    n_clusters = 4
    nb_pobs = 100 # Target for pseudo observations
    r = np.array([2, 1])
    numobs = len(y)
    k = [n_clusters]
    
    seed = 1
    init_seed = 2
        
    eps = 1E-05
    it = 10
    maxstep = 100
    
    nb_runs = 4
    
    for run_idx in range(nb_runs):
        #===========================================#
        # MI2AMI initialisation
        #===========================================# 
        
        init = dim_reduce_init(complete_y, n_clusters, k, r, nj, var_distrib, seed = None,\
                                      use_famd=False)
        out = MI2AMI(y, n_clusters, r, k, init, var_distrib, nj, nan_mask,\
                     nb_pobs, it, eps, maxstep, seed, dm = dm, perform_selec = False)
        
        completed_y = pd.DataFrame(out['completed_y'], columns = full_pima.columns)
        
        #===========================================#
        # MI2AMI full
        #===========================================# 
        
        dm2 = gower_matrix(completed_y, cat_features = cat_features) 
        
        init2 = dim_reduce_init(completed_y.astype(dtype), n_clusters, k, r, nj, var_distrib, seed = None,\
                                      use_famd=False)
            
        out2 = MI2AMI(completed_y, n_clusters, r, k, init2, var_distrib, nj, nan_mask,\
                     nb_pobs, it, eps, maxstep, seed, dm = dm2, perform_selec = False)
        
        completed_y2 = pd.DataFrame(out2['completed_y'], columns = full_pima.columns)
        #completed_y2 = completed_y2.astype(float)
        
        #================================================================
        # Inverse transform both datasets
        #================================================================
        
        for j, colname in enumerate(y.columns):
            if colname in le_dict.keys():
                completed_y[colname] = le_dict[colname].inverse_transform(completed_y[colname].astype(int))
                completed_y2[colname] = le_dict[colname].inverse_transform(completed_y2[colname].astype(int))
        
        assert np.nansum(np.abs(completed_y[~nan_mask].astype(float) - full_pima[~nan_mask].astype(float))) == 0
        assert np.nansum(np.abs(completed_y2[~nan_mask].astype(float) - full_pima[~nan_mask].astype(float))) == 0
        
        # Mimick the format
        completed_y['Outcome'] = completed_y['Outcome'].astype(int)
        completed_y2['Outcome'] = completed_y2['Outcome'].astype(int)
        
        
        #completed_y2.to_csv(res + 'Run' + str(run_idx) + '/imp' + dataset_name + '.csv', index = False)
            
        #================================================================
        # Diagnostic
        #================================================================
        
        ## RMSE and PFC
        completed = {'init': completed_y, 'full': completed_y2}
        error = pd.DataFrame(columns = full_pima.columns, index = completed.keys()) 
        
        # TO DO: ADD weights ?
        for j, col in enumerate(full_pima.columns):
            for method in completed.keys():
                # Fetch the true data
                true = full_pima[nan_mask[col]][col]
                imputed = completed[method][nan_mask[col]][col]
                
                if var_distrib[j] in ['continuous', 'ordinal', 'binomial']: # NRMSE
                    error.loc[method, col] = np.sqrt(np.mean((true - imputed) ** 2) / full_pima[col].var())
                else:
                    error.loc[method, col] = (true.astype(int)!= imputed.astype(int)).mean()
        
            
        # Add Gower
        for method in completed.keys():
            gow = []
            for j, col in enumerate(full_pima.columns):
                # Fetch the true data
                true = full_pima[nan_mask[col]][col]
                imputed = completed[method][nan_mask[col]][col]
                
                if var_distrib[j] in ['continuous', 'ordinal', 'binomial']: # NRMSE
                    cont_range = true.max() - true.min()
                    gow.append(np.mean(np.abs(true - imputed))/cont_range)
                else:
                    gow.append((true.astype(int) != imputed.astype(int)).mean())
            error.loc[method, 'gow'] = np.mean(gow)
        
        #error.T[['full']].T.to_csv(res + 'Run' + str(run_idx) + '/res' + dataset_name + '.csv', index = False)

#=============================
# Comparing associations structure
#=============================

import seaborn as sns
from dython.nominal import compute_associations, associations
from sklearn.metrics.pairwise import cosine_similarity

original_assoc = compute_associations(full_pima, nominal_columns = cat_features)

associations(full_pima, nominal_columns = cat_features)

Ez = out2['Ez.y']
vc = vars_contributions(completed_y2, Ez, assoc_thr = 0.0, \
                       title = 'Contribution of the variables to the latent dimensions',\
                       storage_path = None)

assoc = cosine_similarity(vc, dense_output=True)

labels = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'D.P. Function', 'Age', 'Outcome']

fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize = (12,10)) 
cbar_ax = fig.add_axes([.91, .3, .03, .4])




sns.heatmap(original_assoc.abs(), ax=axn[0],
            cbar=0 == 0,
            vmin=0, vmax=1,
            cbar_ax=None if 0 else cbar_ax,\
            square = True, xticklabels=labels,\
            yticklabels=labels, cmap = 'YlGn')
    
axn[0].set_title('Original association matrix of the full data')
    
sns.heatmap(assoc, ax=axn[1],
            cbar=1 == 0,
            vmin=0, vmax=1,
            cbar_ax=None if 1 else cbar_ax,\
            square = True, xticklabels=labels,\
            yticklabels=labels, cmap = 'YlGn')
    
axn[1].set_title('Estimated association matrix on the imputed data')

fig.tight_layout(rect=[0, 0, .9, 1])
plt.show()


dm3 = gower_matrix(full_pima, cat_features = cat_features) 

init3 = dim_reduce_init(full_pima.astype(dtype), n_clusters, k, r, nj, var_distrib, seed = None,\
                              use_famd=False)
    
out3 = MI2AMI(full_pima, n_clusters, r, k, init2, var_distrib, nj, np.zeros_like(full_pima),\
             nb_pobs, it, eps, maxstep, seed, dm = dm2, perform_selec = False)

completed_y3 = pd.DataFrame(out3['completed_y'], columns = full_pima.columns)

vc3 = vars_contributions(completed_y3, out3['Ez.y'], assoc_thr = 0.0, \
                       title = 'Contribution of the variables to the latent dimensions',\
                       storage_path = None)

assoc3 = cosine_similarity(vc3, dense_output=True)


fig, axn = plt.subplots(1, 2, sharex=True, sharey=True, figsize = (12,10)) 
cbar_ax = fig.add_axes([.91, .3, .03, .4])


sns.heatmap(assoc, ax=axn[0],
            cbar=0 == 0,
            vmin=0, vmax=1,
            cbar_ax=None if 0 else cbar_ax,\
            square = True, xticklabels=labels,\
            yticklabels=labels, cmap = 'YlGn')
    
axn[0].set_title('Latent association matrix of the full data')
    
sns.heatmap(assoc3, ax=axn[1],
            cbar=1 == 0,
            vmin=0, vmax=1,
            cbar_ax=None if 1 else cbar_ax,\
            square = True, xticklabels=labels,\
            yticklabels=labels, cmap = 'YlGn')
    
axn[1].set_title('Latent association matrix on the imputed data')

fig.tight_layout(rect=[0, 0, .9, 1])
plt.show()


np.mean(np.abs(assoc3 - assoc) / assoc)

#================================================================
## Variables Modality Analysis
#================================================================
for col in full_pima.columns:
    full_pima[col].hist()
    plt.title(col)
    plt.show()

#================================================================
## Diagnostic plots
#================================================================

fun = np.array(out['fun'])
fun2 = np.where(fun > 3, 3, fun)
zz = np.stack(out2['zz'])

plt.hist(fun2, bins = 50)  

# Plot of the z used to impute with respect to the distance bewteen y and y imputed
plt.scatter(zz[:,0], zz[:,1], c=fun2)
plt.title('All latent draws z for missing y')
plt.colorbar()
plt.show()


xmin, ymin = out2['Ez.y'].min(0)
xmax, ymax = out2['Ez.y'].max(0)

for var in full_pima.columns:
    obs_with_nan = y.loc[nan_mask.any(1)].reset_index(drop = True)
    nb_nan_per_obs = obs_with_nan.isna().sum(1)
    var_missing_idx = obs_with_nan[obs_with_nan[var].isna()].index
    fun_nan = fun2[var_missing_idx]
    true = full_pima[nan_mask[var]][var]
    imputed = completed_y2[nan_mask[var]][var]
    errors = np.abs(true - imputed).astype(float)
    
    #plt.scatter(fun_nan, errors)
    #plt.title(var + ' Errors in the latent space')
    #plt.show()

    # Error values with respect to the position in the latent space
    plt.scatter(zz[var_missing_idx,0], zz[var_missing_idx,1], c=errors.astype(float), cmap='viridis')
    plt.colorbar()
    
    plt.axhline(ymin)
    plt.axhline(ymax)
    plt.axvline(xmin)
    plt.axvline(xmax)
    
    plt.title(var + ' Errors in the latent space')
    plt.show()

# z vs other quantities
var = 'BloodPressure'

obs_with_nan = y.loc[nan_mask.any(1)].reset_index(drop = True)
nb_nan_per_obs = obs_with_nan.isna().sum(1)
var_missing_idx = obs_with_nan[obs_with_nan[var].isna()].index
true = full_pima[nan_mask[var]][var]
imputed = completed_y2[nan_mask[var]][var]
errors = np.abs(true - imputed).astype(float)

# Imputed vs True missing distribution
plt.hist(true, bins = 20) 
plt.title('True' + var)
plt.show()
plt.hist(imputed, bins = 20)
plt.title('Imputed' + var)
plt.show()

# Impution values with respect to the position in the latent space
plt.scatter(zz[var_missing_idx,0], zz[var_missing_idx,1], c=imputed.astype(float), cmap='viridis')
plt.colorbar()
plt.title('Impution values with respect to the position in the latent space')

# True values with respect to the position in the latent space
plt.scatter(zz[var_missing_idx,0], zz[var_missing_idx,1], c=true.astype(float), cmap='viridis')
plt.colorbar()
plt.title('True values with respect to the position in the latent space')


# Error values with respect to the position in the latent space
plt.scatter(zz[var_missing_idx,0], zz[var_missing_idx,1], c=errors.astype(float), cmap='viridis')
plt.colorbar()
plt.title('Error values with respect to the position in the latent space')


# Plot of the error as a function of the number of values to impute
errorss = [errors[(nb_nan_per_obs[var_missing_idx] == i).to_list()].median() for i in range(1, 6)]
plt.scatter(range(1,6), errors, cmap='viridis')

# True vs error
for true_value in list(set((true))):
    plt.scatter(true_value, errors[true == true_value].median(), color = 'blue')


#====================*
# Ez|y mapping
#====================*

## Look for the z and y mapping
complete_y = complete_y.reset_index(drop = True)
Ez = out2['Ez.y']

## Non-continuous variables
var = 'Outcome'
fig, ax = plt.subplots()
for g in np.unique(full_pima[var]):
    ix = np.where(completed_y2[var] == g)
    ax.scatter(Ez[ix, 0], Ez[ix, 1], label = g, s = 7)
ax.legend()
ax.set_title(var + ' zz')
plt.show()

plt.scatter(Ez[:,0], Ez[:,1], c = full_pima['Pregnancies'].astype(float))
plt.colorbar()
plt.show()


comb = (complete_y['WifeEduc'] == 3.0) & (complete_y['WifeRelig'] == 1.0) & (complete_y['HusbEduc'] == 3.0) 
fig, ax = plt.subplots()
for g in np.unique(comb):
    ix = np.where(comb == g)
    ax.scatter(zz[ix, 0], zz[ix, 1], label = g, s = 7)
ax.legend()
ax.set_title('WifeEduc == 3 & HusbEduc == 3 & WifeRelig == 1')
plt.show()

vars_contributions(completed_y2, Ez, assoc_thr = 0.0, \
                       title = 'Contribution of the variables to the latent dimensions',\
                       storage_path = None)
    
obs_representation(out2['classes'], Ez, storage_path = None)
  


from dython.nominal import associations
associations(full_pima, nominal_columns=cat_features, clustering = True)


full_pima[out2['classes'] == 3]['Outcome'].hist(bins = 20) 
full_pima['Pregnancies'].hist()
