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
from sklearn.preprocessing import MinMaxScaler
#import seaborn as sns

###############################################################################
######################         Adult data          ############################
###############################################################################

res_folder = 'MIAMI/Results/structure/'

#===========================================#
# Model Hyper-parameters
#===========================================#

n_clusters = 3
r = np.array([2, 1])
k = [n_clusters]

seed = 1
init_seed = 2
    
# !!! Changed eps
eps = 1E-05
it = 50
maxstep = 100
nb_pobs = 100
        




    
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

import json
with open("optimal_run.json") as f:
    real = json.load(f)

# from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

brace_dict = {    
    "NONE": 0,
    "H": 1,
    "Z": 2,
    "IZ": 3,
    "ZH": 4,
    "IZH": 5,
    "K": 6,
    "X": 7,
    "XH": 8,
}

brace_dict_inv = dict(zip(brace_dict.values(), brace_dict.keys()))

N_BRACES = len(brace_dict)
def encode(d, max_layers, one_hot=True, native=False):
    basics = [d["legs"], d["total_height"], d["radius_bottom"], d["radius_top"], d["n_layers"]]
    
    # fill design's braces according to max_layers with dummies ("NONE")
    braces = d["connection_types"]
    if native:
        braces = np.array([b for b in braces] + ["NONE"] * (max_layers - 1 - len(braces)))
    else:
        braces = np.array([brace_dict[b] for b in braces] + [brace_dict["NONE"]] * (max_layers - 1 - len(braces)))
        if one_hot:
            braces = get_one_hot(braces, N_BRACES)
    
    # fill design's layer_heights according to max_layers with dummies
    layer_heights = d["layer_heights"]
    layer_heights = np.array(layer_heights + [d["total_height"]] * (max_layers - 2 - len(layer_heights))) / d["total_height"]
    
    # return a flat encoding
    return np.array([*basics, *braces.flatten(), *layer_heights])

def get_cols(d, max_layers, one_hot=True, native=False):
    
    # fill design's braces according to max_layers with dummies ("NONE")
    braces = d["connection_types"]
    if native:
        braces = np.array([b for b in braces] + ["NONE"] * (max_layers - 1 - len(braces)))
    else:
        braces = np.array([brace_dict[b] for b in braces] + [brace_dict["NONE"]] * (max_layers - 1 - len(braces)))
        if one_hot:
            braces = get_one_hot(braces, N_BRACES)
    
    # fill design's layer_heights according to max_layers with dummies
    layer_heights = d["layer_heights"]
    layer_heights = np.array(layer_heights + [d["total_height"]] * (max_layers - 2 - len(layer_heights))) / d["total_height"]
    transformed_columns = ["legs", "total_height", "radius_bottom", "radius_top", "n_layers"] + ["brace" + str(i) for i in range(len(braces.flatten()))] + ["layer_height" + str(i) for i in range(len(layer_heights))]
    
    # return a flat encoding
    return transformed_columns, ["brace" + str(i) for i in range(len(braces.flatten()))]

max_layers = max([d["n_layers"] for d in real])
encodings_real = [encode(d, max_layers, one_hot=False, native=True) for d in real]
transformed_columns, brace_cols = get_cols(real[0], max_layers, one_hot=False)

train = pd.DataFrame(encodings_real, columns=transformed_columns)
nominal_features = brace_cols
ordinal_features = ["n_layers", "legs"]


continuous_features = list(set(transformed_columns) - set(nominal_features) - set(ordinal_features))

train[ordinal_features] = train[ordinal_features].astype("int")
train[continuous_features] = train[continuous_features].astype("float")


train = train.infer_objects()

NUMBER_OBSERVATIONS = -1
train = train.iloc[:NUMBER_OBSERVATIONS, :]


numobs = len(train)
print("Running with", numobs, "observations!!!!")

#*****************************************************************
# Formating the data
#*****************************************************************
# 
unique_counts = train.nunique()
# select columns with variance equal to 0
constant_cols = unique_counts[unique_counts == 1].index
train.drop(constant_cols, axis=1, inplace=True)
print("Dropped", constant_cols, "because of 0 variance")

var_distrib = []     
# Encode categorical datas
for colname, dtype, unique in zip(train.columns, train.dtypes, train.nunique()):
    print(colname, dtype, unique)
    if unique < 2:
        print("Dropped", colname, "because of 0 var")
        train.drop(colname, axis=1, inplace=True)
        continue

    if dtype==int or dtype == "object" and unique==2:
        #bool
        var_distrib.append('bernoulli')
        le = LabelEncoder()
        # Convert them into numerical values               
        train[colname] = le.fit_transform(train[colname]) 
        le_dict[colname] = deepcopy(le)
    elif dtype==int and unique > 2:
        # ordinal
        var_distrib.append('ordinal')
        le = LabelEncoder()
        # Convert them into numerical values               
        train[colname] = le.fit_transform(train[colname]) 
        le_dict[colname] = deepcopy(le)
    elif dtype == "object":
        var_distrib.append('categorical')
        le = LabelEncoder()
        # Convert them into numerical values               
        train[colname] = le.fit_transform(train[colname]) 
        le_dict[colname] = deepcopy(le)
    elif dtype == float:
        var_distrib.append('continuous')
    
var_distrib = np.array(var_distrib)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()
float_cols = train.select_dtypes(include=['float']).columns
# Normalize the continuous features
train.loc[:, float_cols] = scaler.fit_transform(train.loc[:, float_cols])

nj, nj_bin, nj_ord, nj_categ = compute_nj(train, var_distrib)

nb_cont = np.sum(var_distrib == 'continuous')     

p = train.shape[1]
        
# Feature category (cf)
dtype = {train.columns[j]: dtypes_dict[var_distrib[j]] for j in range(p)}

train = train.astype(dtype, copy=True)
numobs = len(train)

authorized_ranges = np.expand_dims(np.stack([[-np.inf,np.inf] for var in var_distrib]).T, 1)

print(len(train.columns))
print(len(var_distrib))
print(train.dtypes)


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

#================================================================
# Inverse transform the datasets
#================================================================
# pred_trans = pred.copy()
# for j, colname in enumerate(train.columns):
#     if colname in le_dict.keys():
#         pred_trans[colname] = le_dict[colname].inverse_transform(pred[colname].astype(int))
    
# pred_trans.loc[:, var_distrib == 'continuous'] = pred_trans.loc[:, var_distrib == 'continuous'].round(0)

print("Saved to", res_folder + 'preds.csv')
# Store the predictions
pred.to_csv('preds.csv', index = False)
#break
  
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
