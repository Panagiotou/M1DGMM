# -*- coding: utf-8 -*-
"""
Created on Mon April 29 13:25:11 2020

@author: rfuchs
"""

import os 

# os.chdir('C:/Users/rfuchs/Documents/GitHub/M1DGMM')

import pandas as pd
from copy import deepcopy
from gower import gower_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 

from MIAMI import MIAMI
from init_params import dim_reduce_init
from data_preprocessing import compute_nj

import autograd.numpy as np
#from table_evaluator import TableEvaluator
from scipy.spatial.distance import pdist, squareform

#import seaborn as sns

###############################################################################
######################         Adult data          ############################
###############################################################################

res_folder = 'MIAMI/Results/Adult/'

if not os.path.exists(res_folder):
    os.makedirs(res_folder)
#===========================================#
# Model Hyper-parameters
#===========================================#

n_clusters = 4
r = np.array([2, 1])
k = [n_clusters]

seed = 1
init_seed = 2
    
# !!! Changed eps
eps = 1E-05
it = 50
maxstep = 100
nb_pobs = 1
        

var_distrib = np.array(['continuous', 'categorical', 'continuous',\
                        'ordinal', 'categorical', 'categorical', 'categorical',\
                        'categorical', 'bernoulli', 'ordinal', 'ordinal',\
                        'continuous', 'categorical', 'bernoulli']) 

# Plotting utilities
varnames = np.array(['age', 'workclass', 'fnlwgt',\
            'education.num', 'marital.status', 'occupation', 'relationship',\
            'race', 'sex', 'capital.gain', 'capital.loss',\
            'hours.per.week', 'native.country', 'income'])

    
dtypes_dict = {'continuous': float, 'categorical': str, 'ordinal': int,\
              'bernoulli': int, 'binomial': int}
    
    

#===========================================#
# Importing data
#===========================================#

os.chdir(res_folder)
inf_nb = 1E12

sub_design = "bivariate"

# acceptance_rate =
le_dict = {}

train_filepath = 'adult.csv'


train = pd.read_csv(train_filepath, sep = ',')
train = train.infer_objects()



# Delete the missing values 
train = train.loc[~(train == '?').any(axis=1)]

NUMBER_OBSERVATIONS = -1
train = train.iloc[:NUMBER_OBSERVATIONS, :]


numobs = len(train)
print("Running with", numobs, "observations!!!!")
# !!! Hack to remove
del(train['education'])
p = train.shape[1]

new_names_dict = dict(zip(train.columns, varnames))

train = train.rename(columns=new_names_dict)
#***************************************************************************
# Invert the order of the columns so that age is no more the first bernoulli
#***************************************************************************
'''
train[['age', 'workclass', 'fnlwgt', 'education.num', 'marital.status',
        'occupation', 'relationship', 'race', 'capital.gain',
        'capital.loss', 'hours.per.week', 'native.country', 'income', 'sex']]


var_distrib = np.array(['continuous', 'categorical', 'continuous',\
            'ordinal', 'categorical', 'categorical', 'categorical',\
            'categorical', 'ordinal', 'ordinal',\
            'continuous', 'categorical', 'bernoulli', 'bernoulli']) 
'''
                    
p_new = len(var_distrib)
cat_features = np.logical_or(var_distrib == 'categorical', var_distrib == 'ordinal')


#*****************************************************************
# Formating the data
#*****************************************************************
                
# Encode categorical datas
for col_idx, colname in enumerate(train.columns):
    if var_distrib[col_idx] == 'categorical': 
        le = LabelEncoder()

        # Convert them into numerical values               
        train[colname] = le.fit_transform(train[colname]) 
        le_dict[colname] = deepcopy(le)

    
# Encode binary data
for col_idx, colname in enumerate(train.columns):
    le = LabelEncoder()

    if var_distrib[col_idx] == 'bernoulli': 
        train[colname] = le.fit_transform(train[colname])
        le_dict[colname] = deepcopy(le)

# Encode ordinal data, modalities have been sorted (at best)
            
ord_le = LabelEncoder()
train['education.num'] = ord_le.fit_transform(train['education.num'])
le_dict['education.num'] = deepcopy(ord_le)

# Encode capital.gain and capital.loss and capital.gain as ordinal variables
for col in ['capital.gain', 'capital.loss']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    le_dict[col] = deepcopy(le)

nj, nj_bin, nj_ord, nj_categ = compute_nj(train, var_distrib)
nb_cont = np.sum(var_distrib == 'continuous')        
        
# Feature category (cf)
dtype = {train.columns[j]: dtypes_dict[var_distrib[j]] for j in range(p)}

train = train.astype(dtype, copy=True)
numobs = len(train)



#*****************************************************************
# Sampling rules
#*****************************************************************    
authorized_ranges = np.expand_dims(np.stack([[-np.inf,np.inf] for var in var_distrib]).T, 1)

if sub_design == 'bivariate':
    # Want to sample only women of more than 60 years old
    authorized_ranges[:,0, 0] = [60, 100]  # Of more than 60 years old

    # Keep only women
    sex_idx = np.argmax(varnames == 'sex')
    women_idx = np.argmax(le_dict['sex'].classes_ == 'Female')
    authorized_ranges[:,0, sex_idx] = [women_idx, women_idx] # Only women
    
elif sub_design == 'trivariate':
    # Want to sample only women of more than 60 years old that are widowed
    authorized_ranges[:,0, 0] = [60, 100]  # Of more than 60 years old
    
    # Keep only women
    sex_idx = np.argmax(varnames == 'sex')
    women_idx = np.argmax(le_dict['sex'].classes_ == 'Female')
    authorized_ranges[:,0, sex_idx] = [women_idx, women_idx] # Only women

    # Keep only widows
    marital_idx = np.argmax(varnames == 'marital.status')                
    widowed_idx = np.argmax(le_dict['marital.status'].classes_ == 'Widowed')
    authorized_ranges[:,0, marital_idx] = [widowed_idx, widowed_idx] # Only widowed
else:
    raise RuntimeError('Not implemented yet') 
    

#*****************************************************************
# Run MIAMI
#*****************************************************************

print("Initialize dimensionality reduction")    
init, transformed_famd_data  = dim_reduce_init(train, n_clusters, k, r, nj, var_distrib, seed = None,\
                                use_famd=True)

print("Computing distance matrix")
# Defining distances over the features
# dm = gower_matrix(train, cat_features = cat_features) 

# Compute the pairwise distances between all elements in my_array
distances = pdist(transformed_famd_data)

# Convert the pairwise distances to a square matrix
dm = squareform(distances)
print("Training")
out = MIAMI(train, n_clusters, r, k, init, var_distrib, nj, authorized_ranges, nb_pobs, it,\
                eps, maxstep, seed, perform_selec = False, dm = dm, max_patience = 0)
# print(out)
print('MIAMI has kept one observation over', round(1 / out['share_kept_pseudo_obs']),\
        'observations generated')
    
acceptance_rate = out['share_kept_pseudo_obs']
pred = pd.DataFrame(out['y_all'], columns = train.columns) 

# #================================================================
# # Inverse transform the datasets
# #================================================================

# for j, colname in enumerate(train.columns):
#     if colname in le_dict.keys():
#         pred[colname] = le_dict[colname].inverse_transform(pred[colname].astype(int))
    
# pred.loc[:, var_distrib == 'continuous'] = pred.loc[:, var_distrib == 'continuous'].round(0)

            
# # Store the predictions
# pred.to_csv('preds.csv', index = False)
# #break
  
# zz = out["zz"]

# z2 = np.vstack([zzz for zzz in zz if len(zzz) >0])
# plt.scatter(z2[:,0], z2[:,1])
# # x1,y1 = polygon.exterior.coords.xy
# # plt.plot(x1,y1)


# # Compare woman, 60+ y.o and people presenting both modalities
# zz = np.concatenate(out['zz'])

# woman_idx = train['sex'] == 0
# age_idx = train['age'] >= 60
# bivariate_idx = woman_idx & age_idx

# fig, ax = plt.subplots(figsize = (9, 9))
# ax.scatter(out['Ez.y'][woman_idx,0], out['Ez.y'][woman_idx,1], c='blue', label = '(Train set) women')
# ax.scatter(out['Ez.y'][age_idx,0], out['Ez.y'][age_idx,1], c='red', label = '(Train set) 60+ years old')
# ax.scatter(zz[:,0], zz[:,1], c='darkgreen', label = '(MIAMI) women 60+ y.o.')
# #plt.title('Latent representation of women and 60+ y.o. individuals from the train set and generated by MIAMI')
# plt.legend()
# plt.show()
