# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:24:43 2022

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

from sklearn.metrics.pairwise import cosine_similarity
from dython.nominal import associations, compute_associations

import autograd.numpy as np

res = 'C:/Users/rfuchs/Documents/These/Stats/MIAMI/Missing_data/MI2AMI/'


from oversample import stat_cont, stat_bin, stat_categ, stat_ord

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


var_distrib = np.array(['continuous', 'ordinal', 'ordinal', 'continuous',\
                        'bernoulli', 'bernoulli', 'categorical', 'ordinal',\
                        'bernoulli', 'categorical'])
    
    
#===========================================#
# Formating the data
#===========================================#

#le_dict = {}
nan_mask = full_contra.isnull()

# Encode categorical datas
for col_idx, colname in enumerate(full_contra.columns):
    if var_distrib[col_idx] == 'categorical':
        le = LabelEncoder()
        full_contra[colname] = le.fit_transform(full_contra[colname])
        #le_dict[colname] = deepcopy(le)

# Encode binary data
for col_idx, colname in enumerate(full_contra.columns):
    if var_distrib[col_idx] == 'bernoulli': 
        le = LabelEncoder()
        full_contra[colname] = le.fit_transform(full_contra[colname])
        #le_dict[colname] = deepcopy(le)
        
# Encode ordinal data
for col_idx, colname in enumerate(full_contra.columns):
    if var_distrib[col_idx] == 'ordinal': 
        le = LabelEncoder()
        full_contra[colname] = le.fit_transform(full_contra[colname])
        #le_dict[colname] = deepcopy(le)
           
#y = y.where(~nan_mask, np.nan)

nj, nj_bin, nj_ord, nj_categ = compute_nj(full_contra, var_distrib)
nb_cont = np.sum(var_distrib == 'continuous')

p_new = full_contra.shape[1]

# Feature category (cf)
dtype = {full_contra.columns[j]: dtypes_dict[var_distrib[j]] for j in range(p_new)}
full_contra = full_contra.astype(dtype, copy=True)

# Feature category (cf)
cat_features = var_distrib == 'categorical'

# Defining distance matrix
dm3 = gower_matrix(full_contra, cat_features = cat_features) 

#===========================================#
# Hyperparameters
#===========================================# 

n_clusters = 2
nb_pobs = 100 # Target for pseudo observations
r = np.array([2, 1])
numobs = len(full_contra)
k = [n_clusters]

seed = 1
init_seed = 2
    
eps = 1E-05
it = 10
maxstep = 100

#===========================================#
# MI2AMI initialisation
#===========================================# 

init_full = dim_reduce_init(full_contra, n_clusters, k, r, nj, var_distrib, seed = None,\
                              use_famd=True)
out_full = MI2AMI(full_contra.astype(float), n_clusters, r, k, init_full, var_distrib, nj, nan_mask,\
             nb_pobs, it, eps, maxstep, seed, dm = dm3, perform_selec = False)

completed_y3 = pd.DataFrame(out_full['completed_y'].round(0), columns = full_contra.columns)

#===========================================#
# Comparison
#===========================================# 

#======================================
# mu_s
#======================================

out['mu'][0]
out2['mu'][0]
out_full['mu'][0]

#======================================
# Lambda
#======================================

out['lambda_cont']
out2['lambda_cont']
out_full['lambda_cont']

#======================================
# Mu @ lambda continuous
#======================================

for i in range(k[0]):
    print([stat_cont(out2['lambda_cont'], out2['mu'][0][i].T) *\
          completed_y2.iloc[:, var_distrib == "continuous"].std()])
        
for i in range(k[0]):
    print(stat_bin(out2['lambda_bin'], out2['mu'][0][i].T, nj_bin))
        
print(stat_cont(out['lambda_cont'], np.array([[40.0, 0.0]])) *\
      complete_y.iloc[:, var_distrib == "continuous"].std())


stat_cont(out_full['lambda_cont'], out_full['mu'][0][0].T) * full_contra.iloc[:, var_distrib == "continuous"].std()
stat_cont(out_full['lambda_cont'], out_full['mu'][0][1].T) * full_contra.iloc[:, var_distrib == "continuous"].std()

# Verification par groupe
for group in [0,1]:
    print(complete_y.loc[out['classes'] == group, 'WifeAge'].mean())

for group in [0,1]:
    print(completed_y2.loc[out2['classes'] == group, 'NbChild'].mean())

for group in [0,1,2, 3]:
    print(full_contra.loc[out_full['classes'] == group, 'NbChild'].mean())

#======================================
# Mu @ lambda binomial
#======================================
stat_bin(out['lambda_bin'], out['mu'][0][0].T, nj_bin)[0] 
stat_bin(out['lambda_bin'], out['mu'][0][1].T, nj_bin)[0] 



#======================================
# Variables contribution
#======================================

# !!! TO DO: A comparer avec la vraie matrice d'association

# Vars contributions
vc = vars_contributions(complete_y, out['Ez.y'], assoc_thr = 0.0, \
                       title = 'Contribution of the variables to the latent dimensions',\
                       storage_path = None)
s = cosine_similarity(vc, dense_output=True)

vc2 = vars_contributions(completed_y2, out2['Ez.y'], assoc_thr = 0.0, \
                       title = 'Contribution of the variables to the latent dimensions',\
                       storage_path = None)
s2 = cosine_similarity(vc2, dense_output=True)

    
vc_full = vars_contributions(full_contra, out_full['Ez.y'], assoc_thr = 0.0, \
                       title = 'Contribution of the variables to the latent dimensions',\
                       storage_path = None)

s_full = cosine_similarity(vc_full, dense_output=True)

# Compare the representation between full and complete
idx = 0
fig, ax = plt.subplots(figsize = (4,4))
ax.scatter(s2[idx], s_full[idx])
plt.title(full_contra.columns[idx])

for i, txt in enumerate(full_contra.columns):
    ax.annotate(txt, (s2[idx][i], s_full[idx][i]))
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])

# Compare the representation between completed cosine similarity and original associations
associations(complete_y.astype(float), nominal_columns = cat_features)
associations(completed_y.astype(float), nominal_columns = cat_features)
associations(full_contra.astype(float), nominal_columns = cat_features)

assoc = compute_associations(full_contra.astype(float), nominal_columns = cat_features).values     


idx = 0
fig, ax = plt.subplots(figsize = (4,4))
ax.scatter(assoc[idx], s_full[idx])
plt.title(full_contra.columns[idx])

for i, txt in enumerate(full_contra.columns):
    ax.annotate(txt, (assoc[idx][i], s_full[idx][i]))
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
  
#======================================
# Ez.y
#======================================
var = 'WifeAge'

plt.scatter(out['Ez.y'][:,0], out['Ez.y'][:,1], c=complete_y[var].astype(float), cmap='viridis')
plt.colorbar()
plt.title(var + ' init')
plt.show()

plt.scatter(out2['Ez.y'][:,0], out2['Ez.y'][:,1], c=full_contra[var].astype(float), cmap='viridis')
plt.colorbar()
plt.title(var + ' completed')
plt.show()


plt.scatter(out_full['Ez.y'][:,0], out_full['Ez.y'][:,1], c=full_contra[var].astype(float), cmap='viridis')
plt.title(var + ' full')
plt.colorbar()
plt.show()

#==================================
# Ez.y per class
#==================================
plt.scatter(out2['Ez.y'][:,0], out2['Ez.y'][:,1], c=out2['classes'])
plt.colorbar()
plt.title(var + ' completed')
plt.show()