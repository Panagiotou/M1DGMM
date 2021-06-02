# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:34:17 2021

@author: rfuchs
"""

import re
import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/M1DGMM')

import pandas as pd
from copy import deepcopy
from gower import gower_matrix
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from data_preprocessing import compute_nj

from m1dgmm import M1DGMM
from init_params import dim_reduce_init

import autograd.numpy as np

os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')
datasets = os.listdir('simulated')

nb_trials = 30


###############################################################################
#######  Simulated data: Find the right number of clusters   ##################
###############################################################################

#===========================================#
# Importing data
#===========================================#

dataset = datasets[-1]

simu1 = pd.read_csv('simulated/result1n500.csv', sep = ';', decimal = ',').iloc[:,1:]
y = simu1.iloc[:,:-1]
labels = simu1.iloc[:,-1]
labels = labels - 1 # Labels starts at 0

y = y.infer_objects()
y_nenc_typed = deepcopy(y.astype(np.object))

numobs = len(y)

n_clusters = len(np.unique(labels))
p = y.shape[1]

#===========================================#
# Formating the data
#===========================================#
var_distrib = np.array(['continuous'] * 10 + ['bernoulli'] * 2 + ['binomial'] * 2
                       + ['categorical'] * 3 + ['ordinal'] * 2) 
    
cat_features = np.logical_or(var_distrib == 'categorical',\
                           var_distrib == 'bernoulli')

# Defining distances over the non encoded features
dm = gower_matrix(y_nenc_typed, cat_features = cat_features) 

# Encode categorical datas
le = LabelEncoder()
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'categorical': 
        y[colname] = le.fit_transform(y[colname])

# Encode ordinal data
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'ordinal': 
        if y[colname].min() != 0: 
            y[colname] = y[colname]  - 1 
    
nj, nj_bin, nj_ord, nj_categ = compute_nj(y, var_distrib)
y_np = y.values
nb_cont = np.sum(var_distrib == 'continuous')

p_new = y.shape[1]

dtype = {y.columns[j]: np.float64 if (var_distrib[j] != 'bernoulli') and \
        (var_distrib[j] != 'categorical') else np.str for j in range(p_new)}

y = y.astype(dtype, copy=True)


#===========================================#
# Running the M1DGMM
#===========================================# 

# Could loop through the numbers of clusters with which the MDGMM is initialized
nb_clusters_start = 7
r = np.array([4, 2])
numobs = len(y)
k = [nb_clusters_start]

seed = 1
init_seed = 2
    
eps = 1E-05
it = 11 # No architecture changes after this point
maxstep = 100

mdgmm_res = pd.DataFrame(columns = ['it_id', 'micro', 'macro', 'silhouette'])

for i in range(nb_trials): 
    prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None,\
                                  use_famd=True)
    
    out = M1DGMM(y_np, 'auto', r, k, prince_init, var_distrib, nj, it,\
                 eps, maxstep, seed, perform_selec = True)
        
    print(len(set(out['classes'])))
    print(out['best_k'])

    
    mdgmm_res = mdgmm_res.append({'it_id': i + 1, 'n_clusters_found': len(set(out['classes']))},\
                                       ignore_index=True)
 
plt.hist(mdgmm_res['n_clusters_found'])
plt.axvline(len(set(labels)), label = 'True number of classes', color = 'orange')
plt.legend(['Number of clusters found', 'True number of classes'])
plt.title('Number of clusters found in the data')
plt.xticks(range(nb_clusters_start + 1))
plt.show()
    
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
    
    dist_min = dm.min()
    dist_max = dm.max()
    dist_range = np.linspace(0, dist_max, 200)

    for linky in linkages: 
        for threshold in dist_range:
            aglo = AgglomerativeClustering(n_clusters = None, affinity ='precomputed',\
                                           linkage = linky, distance_threshold = threshold)
            
            aglo_preds = aglo.fit_predict(dm)
    
            hierarch_res = hierarch_res.append({'dataset': dataset, 'linkage': linky, \
                                'dist_threshold': threshold, 'n_clusters_found':len(set(aglo_preds))},\
                                               ignore_index=True)
 
hierarch_res.to_csv('C:/Users/rfuchs/Documents/These/Experiences/found_cluster_simus/hierarchical.csv',\
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

# scaled data eps = 3.7525 and min_samples = 4  is the best spe
mean_res = dbs_res.groupby(['data','leaf_size', 'eps', 'min_samples'])['n_clusters_found'].mean()
maxs = mean_res.max()

hierarch_res.to_csv('C:/Users/rfuchs/Documents/These/Experiences/found_cluster_simus/dbscan.csv',\
                    index = False)

###############################################################################
################################  Result analysis  ############################
###############################################################################

#===========================================#
# Hierarchical clustering
#===========================================# 

hierarch_res = pd.read_csv('C:/Users/rfuchs/Documents/These/Experiences/found_cluster_simus/hierarchical.csv')
hierarch_res['n_clusters_found'] =  hierarch_res['n_clusters_found'].astype(int)

linkages = ['complete', 'average', 'single']
colors = ['#9467bd', '#2ca02c', '#d62728', 'orange', '#1f77b4']
    
for dataset in datasets: 
    # Plot the results
    data = hierarch_res[hierarch_res['dataset'] == dataset]
    
    for idx, linky in enumerate(linkages): 
        plt.plot(data[data['linkage'] == linky].set_index('dist_threshold')[['n_clusters_found']],\
                 color = colors[idx])
    plt.axhline(len(set(labels)), label = 'True number of classes', color = 'orange', linestyle = 'dashed')
    plt.legend([*['Number of clusters found (' + linky + ')' for linky in linkages] , 'True number of classes'])
    
    plt.yscale('log')
    plt.ylabel('Number of clusters found')
    plt.xlabel('Distance threshold used')
    
    plt.title(dataset[5:-4] + 'Number of clusters found in the data for different threshold distance used')
    plt.show()

###############################################################################
####  Simulated data: Assess the percentage of partition similarity  ##########
###############################################################################

'''
For each dataset
For 30 runs
Look at the similarity between the partitions
ARI score ? 
Look in Selosse
'''