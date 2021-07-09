# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:20:17 2021

@author: rfuchs
"""


import re
import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/M1DGMM')

import pandas as pd
import seaborn as sns
from copy import deepcopy
from gower import gower_matrix
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score


from data_preprocessing import compute_nj

from m1dgmm import M1DGMM
from init_params import dim_reduce_init

import autograd.numpy as np

results_path = 'C:/Users/rfuchs/Documents/These/Experiences/' # Results storage
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')
datasets = os.listdir('simulated')

nb_trials = 30


###############################################################################
#######  Simulated data: Find the right number of clusters   ##################
###############################################################################

#===========================================#
# Importing data
#===========================================#

# Hyper-parameters
nb_clusters_start = 7
r = np.array([4, 2])
numobs = len(y)
k = [nb_clusters_start]

seed = 1
init_seed = 2
    
eps = 1E-05
it = 11 # No architecture changes after this point
maxstep = 100

mdgmm_res = pd.DataFrame(columns = ['dataset', 'it_id', 'r', 'k', \
                                    'best_r', 'best_k', 'n_clusters_found'])

type_detection_regex = {'yC[0-9]\.[0-9]{1,2}': 'continuous', 'yBer[0-9]\.[0-9]{1,2}': 'bernoulli',\
                    'yBin[0-9]\.[0-9]{1,2}': 'binomial' , 'yM[0-9]\.[0-9]{1,2}': 'categorical',\
                    'yOrdi[0-9]\.[0-9]{1,2}': 'ordinal'}

for dataset in datasets: 

    simu = pd.read_csv('simulated/' + dataset, sep = ',', decimal = ',').iloc[:,1:]
    if simu.shape[1] == 0: # The separator is not constant..
        simu = pd.read_csv('simulated/' + dataset, sep = ';', decimal = ',').iloc[:,1:]
    
    y = simu.iloc[:,:-1]
    numobs = len(y)
    p = y.shape[1]

    # Determine the type from the name of the variable (dirty)
    var_distrib = pd.Series(y.columns)
    [var_distrib.replace(regex, var_type, regex = True, inplace = True)\
                   for regex, var_type  in type_detection_regex.items()]
    var_distrib = var_distrib.values
    
    #===========================================#
    # Formating the data
    #===========================================#
  
    # Encode categorical datas
    le = LabelEncoder()
    for col_idx, colname in enumerate(y.columns):
        if var_distrib[col_idx] == 'categorical': 
            y[colname] = le.fit_transform(y[colname]).astype(np.str)
    
    # Encode ordinal data
    for col_idx, colname in enumerate(y.columns):
        if var_distrib[col_idx] == 'ordinal': 
            if y[colname].min() != 0: 
                y[colname] = y[colname]  - 1 
        
    nj, nj_bin, nj_ord, nj_categ = compute_nj(y, var_distrib)
    y_np = y.values
    nb_cont = np.sum(var_distrib == 'continuous')

    cat_features = pd.Series(var_distrib).isin(['categorical', 'bernoulli']).to_list()
    dtype = {y.columns[j]: np.str if cat_features[j] else np.float64 for j in range(p)}
    y = y.astype(dtype, copy=True)

    dm = gower_matrix(y, cat_features = cat_features) 
        
    #===========================================#
    # Running the M1DGMM
    #===========================================# 
        
    for i in range(nb_trials): 
        prince_init = dim_reduce_init(y, nb_clusters_start, k, r, nj, var_distrib, seed = None,\
                                      use_famd=True)
        
        out = M1DGMM(y_np, 'auto', r, k, prince_init, var_distrib, nj, it,\
                     eps, maxstep, seed, perform_selec = True, dm = dm)
                
        mdgmm_res = mdgmm_res.append({'dataset': dataset, 'it_id': i + 1,\
                                      'n_clusters_found': len(set(out['classes'])),\
                                      'r': r, 'k':k,\
                                      'best_r': out['best_r'], 'best_k':out['best_k']},\
                                           ignore_index=True)
            
#===========================================#
# Running the hierarchical clustering
#===========================================# 

hierarch_res = pd.DataFrame(columns = ['dataset', 'linkage', 'dist_threshold', 'n_clusters_found'])
linkages = ['complete', 'average', 'single']

for dataset in datasets: 
    
    #===========================================#
    # Formating the data
    #===========================================#
    
    simu = pd.read_csv('simulated/' + dataset, sep = ',', decimal = ',').iloc[:,1:]
    if simu.shape[1] == 0: # The separator is not constant..
        simu = pd.read_csv('simulated/' + dataset, sep = ';', decimal = ',').iloc[:,1:]
    
    y = simu.iloc[:,:-1]    
    p = y.shape[1]
        
    # Ordinal and continuous are not categorical
    cat_features = [re.search('y[CO]', col) == None for col in y.columns]
    dtype = {y.columns[j]: np.str if cat_features[j] else np.float64 for j in range(p)}
    y = y.astype(dtype, copy=True)

    # Defining distances over the non encoded features
    dm = gower_matrix(y, cat_features = cat_features) 
    
    dist_min = dm[dm>0].min()
    dist_max = dm.max()
    dist_range = np.linspace(dist_min, dist_max, 200)

    for linky in linkages: 
        for threshold in dist_range:
            aglo = AgglomerativeClustering(n_clusters = None, affinity ='precomputed',\
                                           linkage = linky, distance_threshold = threshold)
            
            aglo_preds = aglo.fit_predict(dm)
    
            hierarch_res = hierarch_res.append({'dataset': dataset, 'linkage': linky, \
                                'dist_threshold': threshold, 'n_clusters_found':len(set(aglo_preds))},\
                                               ignore_index=True)
 
hierarch_res.to_csv(results_path + 'find_nclusters/data/Hierarchical/hierarchical.csv',\
                    index = False)

#===========================================#
# Running the DBSCAN clustering
#===========================================#
 
dbs_res = pd.DataFrame(columns = ['dataset', 'it_id', 'data' ,'leaf_size', 'eps',\
                                  'min_samples', 'n_clusters_found'])
        
for dataset in datasets: 
    
    #===========================================#
    # Formating the data
    #===========================================#
    
    simu = pd.read_csv('simulated/' + dataset, sep = ',', decimal = ',').iloc[:,1:]
    if simu.shape[1] == 0: # The separator is not constant..
        simu = pd.read_csv('simulated/' + dataset, sep = ';', decimal = ',').iloc[:,1:]
    
    y = simu.iloc[:,:-1]  
    y_nenc_typed = deepcopy(y.astype(np.object))

    p = y.shape[1]
        
    # Ordinal and continuous are not categorical
    cat_features = [re.search('y[CO]', col) == None for col in y.columns]
    dtype = {y.columns[j]: np.str if cat_features[j] else np.float64 for j in range(p)}
    y = y.astype(dtype, copy=True)

    # Defining distances over the non encoded features
    dm = gower_matrix(y, cat_features = cat_features) 

    # Scale the continuous variables
    cont_features = [re.search('yC', col) != None for col in y.columns]
    y_scale = y.values
    ss = StandardScaler()
    y_scale[:, cont_features] = ss.fit_transform(y_scale[:, cont_features])
            
    lf_size = np.arange(1,6) * 10
    epss = np.linspace(0.01, 5, 5)
    min_ss = np.arange(1, 5)
    data_to_fit = ['scaled', 'gower']
    
    for lfs in lf_size:
        print("Leaf size:", lfs)
        for eps in epss:
            for min_s in min_ss:
                for data in data_to_fit:
                    for i in range(nb_trials):
                        if data == 'gower':
                            dbs = DBSCAN(eps = eps, min_samples = min_s, \
                                         metric = 'precomputed', leaf_size = lfs).fit(dm)
                        else:
                            dbs = DBSCAN(eps = eps, min_samples = min_s, leaf_size = lfs).fit(y_scale)
                            
                        dbs_preds = dbs.labels_                    
                        dbs_res = dbs_res.append({'dataset': dataset, 'it_id': i + 1, 'leaf_size': lfs, \
                                    'eps': eps, 'min_samples': min_s, 'data': data,\
                                        'n_clusters_found': len(set(dbs_preds))},\
                                                 ignore_index=True)

dbs_res.to_csv(results_path + 'find_nclusters/data/DBSCAN/dbscan.csv',\
                    index = False)
