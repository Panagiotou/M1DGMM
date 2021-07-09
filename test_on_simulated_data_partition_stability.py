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


from minisom import MiniSom   
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering



from data_preprocessing import compute_nj

from m1dgmm import M1DGMM
from init_params import dim_reduce_init

import autograd.numpy as np

results_path = 'C:/Users/rfuchs/Documents/These/Experiences/' # Results storage
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')
datasets = os.listdir('simulated')

nb_trials = 30
n_clusters = 4

###############################################################################
####  Simulated data: Assess the percentage of partition similarity  ##########
###############################################################################

r = np.array([5, 2, 1])
k = [n_clusters, 3]

seed = 1
init_seed = 2
    
eps = 1E-02
it = 30 
maxstep = 100

nb_trials = 30

type_detection_regex = {'yC[0-9]\.[0-9]{1,2}': 'continuous', 'yBer[0-9]\.[0-9]{1,2}': 'bernoulli',\
                        'yBin[0-9]\.[0-9]{1,2}': 'binomial' , 'yM[0-9]\.[0-9]{1,2}': 'categorical',\
                        'yOrdi[0-9]\.[0-9]{1,2}': 'ordinal'}

#===========================================#
# M1DGMM
#===========================================# 

for dataset in datasets: 

    simu = pd.read_csv('simulated/' + dataset, sep = ',', decimal = ',').iloc[:,1:]
    if simu.shape[1] == 0: # The separator is not constant..
        simu = pd.read_csv('simulated/' + dataset, sep = ';', decimal = ',').iloc[:,1:]
    
    y = simu.iloc[:,:-1]
    numobs = len(y)
    p = y.shape[1]
    
    mdgmm_res = pd.DataFrame(columns = range(0,nb_trials), index = range(numobs))


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
        prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None,\
                                      use_famd=True)
        
        out = M1DGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it,\
                     eps, maxstep, seed, perform_selec = False, dm = dm)
                
        mdgmm_res.iloc[:,i] = out['classes']
        
        mdgmm_res = mdgmm_res.append({'dataset': dataset, 'it_id': i, 'classes': out['classes']},\
                                           ignore_index=True)
         
    mdgmm_res.to_csv(results_path + 'similar_partition/' + dataset,\
                    index = False, header = False)
    
#===========================================#
# Hierarchical clustering
#===========================================# 

linkages = ['complete', 'average', 'single']

for dataset in datasets: 
    
    #===========================================#
    # Formating the data
    #===========================================#
    
    simu = pd.read_csv('simulated/' + dataset, sep = ',', decimal = ',').iloc[:,1:]
    if simu.shape[1] == 0: # The separator is not constant..
        simu = pd.read_csv('simulated/' + dataset, sep = ';', decimal = ',').iloc[:,1:]
    
    y = simu.iloc[:,:-1] 
    numobs = len(y)
    p = y.shape[1]

    # Ordinal and continuous are not categorical
    cat_features = [re.search('y[CO]', col) == None for col in y.columns]
    dtype = {y.columns[j]: np.str if cat_features[j] else np.float64 for j in range(p)}
    y = y.astype(dtype, copy=True)

    # Defining distances over the non encoded features
    dm = gower_matrix(y, cat_features = cat_features) 
    
    for linky in linkages: 
        hierarch_res = pd.DataFrame(columns = range(0,nb_trials), index = range(numobs))

        for i in range(nb_trials):
            
            aglo = AgglomerativeClustering(n_clusters = 4, affinity ='precomputed',\
                                           linkage = linky)
            
            aglo_preds = aglo.fit_predict(dm)
            hierarch_res.iloc[:,i] = aglo_preds
     
        hierarch_res.to_csv(results_path + 'similar_partition/data/Hierarchical/' +\
                            dataset[:-4] + '_' + linky + '.csv', index = False, header = False)

    
#===========================================#
# DBSCAN clustering
#===========================================# 

for dataset in datasets: 
    
    #===========================================#
    # Formating the data
    #===========================================#
    
    simu = pd.read_csv('simulated/' + dataset, sep = ',', decimal = ',').iloc[:,1:]
    if simu.shape[1] == 0: # The separator is not constant..
        simu = pd.read_csv('simulated/' + dataset, sep = ';', decimal = ',').iloc[:,1:]
    
    y = simu.iloc[:,:-1]  
    numobs = len(y)
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
                    dbs_res = pd.DataFrame(columns = range(0,nb_trials), index = range(numobs))

                    for i in range(nb_trials):
                        if data == 'gower':
                            dbs = DBSCAN(eps = eps, min_samples = min_s, \
                                         metric = 'precomputed', leaf_size = lfs).fit(dm)
                        else:
                            dbs = DBSCAN(eps = eps, min_samples = min_s, leaf_size = lfs).fit(y_scale)
                            
                        dbs_res.iloc[:,i] = dbs.labels_

                    dbs_res.to_csv(results_path + 'similar_partition/data/DBSCAN/dbscan' + \
                                   dataset[:-4] + '_' + str(lfs) + '_' + str(eps) + '_' + str(min_s) + '_' +\
                                       str(data) + '.csv', index = False, header = False)


#===========================================# 
# Partitional algorithm
#===========================================# 
inits = ['Huang', 'Cao', 'random']

for dataset in datasets: 
    
    #===========================================#
    # Formating the data
    #===========================================#
    ss = StandardScaler()
    
    simu = pd.read_csv('simulated/' + dataset, sep = ',', decimal = ',').iloc[:,1:]
    if simu.shape[1] == 0: # The separator is not constant..
        simu = pd.read_csv('simulated/' + dataset, sep = ';', decimal = ',').iloc[:,1:]
    
    y = simu.iloc[:,:-1] 
    numobs = len(y)

    
    cont_features = [re.search('yC', col) != None for col in y.columns]
    y_scale = y.values
    y_scale[:, cont_features] = ss.fit_transform(y_scale[:, cont_features])
    
    for init in inits:
        print(init)
        part_res_modes = pd.DataFrame(columns = range(0,nb_trials), index = range(numobs))
    
        for i in range(nb_trials):
            km = KModes(n_clusters= n_clusters, init=init, n_init=10, verbose=0)
            kmo_labels = km.fit_predict(y_scale)
            
            part_res_modes.iloc[:,i] = kmo_labels
            
            
        part_res_modes.to_csv(results_path + 'similar_partition/data/KMODES/' + \
                                       dataset[:-4] + '_' + init + '.csv',\
                                           index = False, header = False)
        
        
#===========================================# 
# K prototypes
#===========================================# 
inits = ['Huang', 'Cao', 'random']

for dataset in datasets: 
    
    #===========================================#
    # Formating the data
    #===========================================#
    ss = StandardScaler()
    
    simu = pd.read_csv('simulated/' + dataset, sep = ',', decimal = ',').iloc[:,1:]
    if simu.shape[1] == 0: # The separator is not constant..
        simu = pd.read_csv('simulated/' + dataset, sep = ';', decimal = ',').iloc[:,1:]
    
    y = simu.iloc[:,:-1] 
    numobs = len(y)
    
    cont_features = [re.search('yC', col) != None for col in y.columns]
    cat_features = [not(el) for el in cont_features]

    y_scale = y.values
    y_scale[:, cont_features] = ss.fit_transform(y_scale[:, cont_features])
    
    for init in inits:
        print(init)
        part_res_proto = pd.DataFrame(columns = range(0,nb_trials), index = range(numobs))
    
        for i in range(nb_trials):
            km = KPrototypes(n_clusters = n_clusters, init = init, n_init=10, verbose=0)
            kmo_labels = km.fit_predict(y_scale, categorical = np.where(cat_features)[0].tolist())
    
            part_res_proto.iloc[:,i] = kmo_labels

        part_res_proto.to_csv(results_path + 'similar_partition/data/KPROTOTYPES/' + \
                                       dataset[:-4] + '_' + init + '.csv',\
                                           index = False, header = False)
            
        
#****************************
# Neural-network based
#****************************
sigmas = np.linspace(0.001, 3, 5)
lrs = np.linspace(0.0001, 0.5, 10)

for dataset in datasets: 
    
    #===========================================#
    # Formating the data
    #===========================================#
    ss = StandardScaler()
    
    simu = pd.read_csv('simulated/' + dataset, sep = ',', decimal = ',').iloc[:,1:]
    if simu.shape[1] == 0: # The separator is not constant..
        simu = pd.read_csv('simulated/' + dataset, sep = ';', decimal = ',').iloc[:,1:]
    
    y = simu.iloc[:,:-1] 
    numobs = len(y)
    y_scale = y.values
    cont_features = [re.search('yC', col) != None for col in y.columns]

    y_scale[:, cont_features] = ss.fit_transform(y_scale[:, cont_features])
    

    
    for sig in sigmas:
        for lr in lrs:
            som_res = pd.DataFrame(columns = range(0,nb_trials), index = range(numobs))
    
            for i in range(nb_trials):
                #try:
                som = MiniSom(n_clusters, 1, y_scale.shape[1], sigma = sig, learning_rate = lr) # initialization of 6x6 SOM
                som.train(y_scale.astype(float), 100) # trains the SOM with 100 iterations
                som_labels = [som.winner(y_scale.astype(float)[i])[0] for i in range(numobs)]
                som_res.iloc[:,i] = som_labels

                #except TypeError:
                    #som_res.iloc[:,i] = np.nan

                    
            som_res.to_csv(results_path + 'similar_partition/data/SOM/' + \
                                           dataset[:-4] + '_' + str(sig) + '_' + str(lr) + '.csv',\
                                               index = False, header = False)
                
            
            

