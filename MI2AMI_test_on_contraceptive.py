# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:53:30 2021

@author: rfuchs
"""

import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/M1DGMM')

import pandas as pd
from copy import deepcopy
from gower import gower_matrix
import matplotlib .pyplot as plt
from sklearn.preprocessing import LabelEncoder 

from mi2ami import MI2AMI
from init_params import dim_reduce_init
from utilities import vars_contributions
from data_preprocessing import compute_nj#, data_processing

import autograd.numpy as np

res = 'C:/Users/rfuchs/Documents/These/Stats/MIAMI/Missing_data/MI2AMI/'

###############################################################################
###############     Contraceptive    vizualisation      #######################
###############################################################################

dtypes_dict = {'continuous': float, 'categorical': str, 'ordinal': float,\
              'bernoulli': int, 'binomial': int}

#===========================================#
# Importing data
#===========================================#

os.chdir('C:/Users/rfuchs/Documents/These/Stats/MIAMI/Missing_data/Data')

full_contra = pd.read_csv('cmc2.csv', sep = ';')

dataset_name = 'MCAR30'
y = pd.read_csv(dataset_name + '.csv', sep = ';').iloc[:,1:]

var_distrib = np.array(['continuous', 'ordinal', 'ordinal', 'continuous',\
                        'bernoulli', 'bernoulli', 'categorical', 'ordinal',\
                        'bernoulli', 'categorical'])
    
#yy = data_processing(y, var_distrib)
#y.loc[:, var_distrib == 'ordinal'].iloc[:,1].value_counts()

#===========================================#
# Formating the data
#===========================================#

le_dict = {}

nan_mask = y.isnull()
                        
# Encode categorical datas
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'categorical':
        le = LabelEncoder()
        y[colname] = le.fit_transform(y[colname])
        le_dict[colname] = deepcopy(le)

# Encode binary data
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'bernoulli': 
        le = LabelEncoder()
        y[colname] = le.fit_transform(y[colname])
        le_dict[colname] = deepcopy(le)
        
# Encode ordinal data
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'ordinal': 
        le = LabelEncoder()
        y[colname] = le.fit_transform(y[colname])
        le_dict[colname] = deepcopy(le)
           
y = y.where(~nan_mask, np.nan)

nj, nj_bin, nj_ord, nj_categ = compute_nj(full_contra, var_distrib)
nb_cont = np.sum(var_distrib == 'continuous')

p_new = y.shape[1]

# Feature category (cf)
dtype = {y.columns[j]: dtypes_dict[var_distrib[j]] for j in range(p_new)}

full_contra = full_contra.astype(dtype, copy=True)
complete_y = y[~y.isna().any(1)]
complete_y = complete_y.astype(dtype, copy=True)

# Feature category (cf)
cat_features = var_distrib == 'categorical'

# Defining distance matrix
dm = gower_matrix(complete_y, cat_features = cat_features) 

#===========================================#
# Hyperparameters
#===========================================# 

n_clusters = 2
nb_pobs = 100 # Target for pseudo observations
r = np.array([2, 1])
numobs = len(y)
k = [n_clusters]

seed = 1
init_seed = 2
    
eps = 1E-05
it = 10
maxstep = 100

#===========================================#
# MI2AMI initialisation
#===========================================# 

init = dim_reduce_init(complete_y, n_clusters, k, r, nj, var_distrib, seed = None,\
                              use_famd=True)
out = MI2AMI(y, n_clusters, r, k, init, var_distrib, nj, nan_mask,\
             nb_pobs, it, eps, maxstep, seed, dm = dm, perform_selec = False)

completed_y = pd.DataFrame(out['completed_y'].round(0), columns = full_contra.columns)

#*************************************
# Ensure that continuous variables stay in their support
#*************************************

for j, colname in enumerate(completed_y.columns):
    if var_distrib[j] == 'continuous':
        completed_y.loc[completed_y[colname] < y[colname].min(), colname] = y[colname].min()
        completed_y.loc[completed_y[colname] > y[colname].max(), colname] = y[colname].max()

#===========================================#
# MI2AMI full
#===========================================# 

#*************************************
# Defining distance matrix
#*************************************

dm2 = gower_matrix(completed_y, cat_features = cat_features) 

init2 = dim_reduce_init(completed_y.astype(dtype), n_clusters, k, r, nj, var_distrib, seed = None,\
                              use_famd=True)
    
out2 = MI2AMI(completed_y.values, n_clusters, r, k, init2, var_distrib, nj, nan_mask,\
             nb_pobs, it, eps, maxstep, seed, dm = dm2, perform_selec = False)

completed_y2 = pd.DataFrame(out2['completed_y'], columns = full_contra.columns)
completed_y2 = completed_y2.astype(float).round(0)

#*************************************
# Ensure that continuous variables stay in their support
#*************************************

for j, colname in enumerate(completed_y2.columns):
    if var_distrib[j] == 'continuous':
        completed_y2.loc[completed_y2[colname] < y[colname].min(), colname] = y[colname].min()
        completed_y2.loc[completed_y2[colname] > y[colname].max(), colname] = y[colname].max()

#================================================================
# Inverse transform both datasets
#================================================================

for j, colname in enumerate(y.columns):
    if colname in le_dict.keys():
        completed_y[colname] = le_dict[colname].inverse_transform(completed_y[colname].astype(int))
        completed_y2[colname] = le_dict[colname].inverse_transform(completed_y2[colname].astype(int))

assert np.nansum(np.abs(completed_y[~nan_mask].astype(float) - full_contra[~nan_mask].astype(float))) == 0
assert np.nansum(np.abs(completed_y2[~nan_mask].astype(float) - full_contra[~nan_mask].astype(float))) == 0

# Mimick the format
completed_y = completed_y.astype(int)
completed_y2 = completed_y2.astype(int)

completed_y2.to_csv(res + 'imp' + dataset_name + '.csv', index = False)

#================================================================
# Diagnostic
#================================================================

## RMSE and PFC
completed = {'init': completed_y, 'full': completed_y2}
error = pd.DataFrame(columns = full_contra.columns, index = completed.keys()) 

# TO DO: ADD weights ?
for j, col in enumerate(full_contra.columns):
    for method in completed.keys():
        # Fetch the true data
        true = full_contra[nan_mask[col]][col]
        imputed = completed[method][nan_mask[col]][col]
        
        if (var_distrib[j] == 'continuous') & (var_distrib[j] == 'ordinal'): # NRMSE
            print(true)
            error.loc[method, col] = np.sqrt(np.mean((true - imputed) ** 2) / true.var())
        else:
            error.loc[method, col] = (true.astype(int)!= imputed.astype(int)).mean()

cols = ["nrmseAge","nrmseWeduc","nrmseHeduc","nrmesChild","PFCrelig","PFCwork",\
        "PFCoccup","rnmseStand","PFCmedia","PFCcont","Gower"]
    
# Add Gower
for method in completed.keys():
    gow = []
    for j, col in enumerate(full_contra.columns):
        # Fetch the true data
        true = full_contra[nan_mask[col]][col]
        imputed = completed[method][nan_mask[col]][col]
        
        if (var_distrib[j] == 'continuous') & (var_distrib[j] == 'ordinal'): # NRMSE
            cont_range = true.max() - true.min()
            gow.append(np.mean(np.abs(true - imputed))/cont_range)
        else:
            gow.append((true.astype(int) != imputed.astype(int)).mean())
    error.loc[method, 'gow'] = np.mean(gow)

error.columns = cols
error.T[['full']].T.to_csv(res + 'res' + dataset_name + '.csv', index = False)














#================================================================
## Diagnostic plots
#================================================================

plt.scatter(error.loc['init'], error.loc['full'])
plt.plot(np.linspace(0, 1), np.linspace(0, 1), color = 'red')
plt.show()

plt.hist(full_contra[nan_mask['WifeAge']]['WifeAge'], bins = 40) 
plt.hist(completed_y2[nan_mask['WifeAge']]['WifeAge'], bins = 40) 

var = 'NbChild'
plt.scatter(full_contra[nan_mask[var]][var], completed_y2[nan_mask[var]][var])

var = 'Standard'
pd.crosstab(full_contra[nan_mask[var]][var], completed_y2[nan_mask[var]][var],\
            normalize = True, rownames=['True'], colnames = ['Imputed'])



plt.scatter(full_contra[nan_mask.iloc[:,0]]['WifeAge'],\
         completed_y[nan_mask.iloc[:,0]]['WifeAge'])
plt.title('True vs imputed')
plt.show()

out['lambda_cont'][0]
out['best_r']

## Look for the z and y mapping
complete_y = complete_y.reset_index(drop = True)
zz = out['Ez.y']

## Individual variables
var = 'WifeRelig'
fig, ax = plt.subplots()
for g in np.unique(complete_y[var]):
    ix = np.where(complete_y[var] == g)
    ax.scatter(zz[ix, 0], zz[ix, 1], label = g, s = 7)
ax.legend()
ax.set_title(var + ' zz')
plt.show()


comb = (complete_y['WifeEduc'] == 3.0) & (complete_y['WifeRelig'] == 1.0) & (complete_y['HusbEduc'] == 3.0) 
fig, ax = plt.subplots()
for g in np.unique(comb):
    ix = np.where(comb == g)
    ax.scatter(zz[ix, 0], zz[ix, 1], label = g, s = 7)
ax.legend()
ax.set_title('WifeEduc == 3 & HusbEduc == 3 & WifeRelig == 1')
plt.show()

plt.scatter(zz[:,0], zz[:,1], c = complete_y['HusbEduc'].astype(float))


vars_contributions(full_contra, zz, assoc_thr = 0.0, \
                       title = 'Contribution of the variables to the latent dimensions',\
                       storage_path = None)
    
