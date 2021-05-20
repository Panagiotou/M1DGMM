# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:34:17 2021

@author: rfuchs
"""

import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/M1DGMM')

from copy import deepcopy

from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

from gower import gower_matrix
from sklearn.metrics import silhouette_score


from m1dgmm import M1DGMM
from MIAMI import miami
from init_params import dim_reduce_init
from metrics import misc
from data_preprocessing import gen_categ_as_bin_dataset, \
        compute_nj


import autograd.numpy as np
from autograd.numpy.random import uniform


###############################################################################
################   Simulated data    vizualisation      #######################
###############################################################################

#===========================================#
# Importing data
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')

simu1 = pd.read_csv('simulated/result1n500.csv', sep = ';', decimal = ',').iloc[:,1:]
y = simu1.iloc[:,:-1]
labels = simu1.iloc[:,-1]
labels = labels - 1 # Labels starts at 0

y = y.infer_objects()
numobs = len(y)

n_clusters = len(np.unique(labels))
p = y.shape[1]

#===========================================#
# Formating the data
#===========================================#
var_distrib = np.array(['continuous'] * 10 + ['bernoulli'] * 2 + ['binomial'] * 2
                       + ['categorical'] * 3) 
    
# Ordinal data already encoded
y_categ_non_enc = deepcopy(y)
vd_categ_non_enc = deepcopy(var_distrib)

# Encode categorical datas
le = LabelEncoder()
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'categorical': 
        y[colname] = le.fit_transform(y[colname])

    
nj, nj_bin, nj_ord, nj_categ = compute_nj(y, var_distrib)
y_np = y.values
nb_cont = np.sum(var_distrib == 'continuous')

p_new = y.shape[1]


# Feature category (cf)
cf_non_enc = np.logical_or(vd_categ_non_enc == 'categorical', vd_categ_non_enc == 'bernoulli')

# Non encoded version of the dataset:
y_nenc_typed = y_categ_non_enc.astype(np.object)
y_np_nenc = y_nenc_typed.values

# Defining distances over the non encoded features
dm = gower_matrix(y_nenc_typed, cat_features = cf_non_enc) 

dtype = {y.columns[j]: np.float64 if (var_distrib[j] != 'bernoulli') and \
        (var_distrib[j] != 'categorical') else np.str for j in range(p_new)}

y = y.astype(dtype, copy=True)

#===========================================#
# Running the algorithm
#===========================================# 

r = np.array([6, 3])
numobs = len(y)
k = [n_clusters]

seed = 1
init_seed = 2
    
eps = 1E-05
it = 10
maxstep = 100

prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None,\
                              use_famd=True)
m, pred = misc(labels, prince_init['classes'], True) 
print(m)
print(confusion_matrix(labels, pred))
print(silhouette_score(dm, pred, metric = 'precomputed'))

'''
init = prince_init
seed = None
y = y_np
perform_selec = False
os.chdir('C:/Users/rfuchs/Documents/GitHub/M1DGMM')
'''


out = M1DGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it,\
             eps, maxstep, seed, perform_selec = False)
m, pred = misc(labels, out['classes'], True) 
print(m)
print(confusion_matrix(labels, pred))
print(silhouette_score(dm, pred, metric = 'precomputed'))

# Plot the final groups

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

colors = ['green','red', 'blue', 'violet']

fig = plt.figure(figsize=(8,8))
plt.scatter(out["z"][:, 0], out["z"][:, 1], c=pred,\
            cmap=matplotlib.colors.ListedColormap(colors))

plt.show()

fig = plt.figure(figsize=(8,8))
plt.scatter(out["z"][:, 0], out["z"][:, 1], c=labels,\
            cmap=matplotlib.colors.ListedColormap(colors))
plt.show()
