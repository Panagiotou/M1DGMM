# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:35:27 2021

@author: rfuchs
"""

import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/M1DGMM')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import autograd.numpy as np

results_path = 'C:/Users/rfuchs/Documents/These/Experiences/' # Results storage
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')
datasets = os.listdir('simulated')

n_clusters = 4
nb_trials = 30

designs = ['1n500', '1n1000', '1bisn500', '1bisn1000','2n500', '2n1000', '2bisn500', '2bisn1000']
colors = ['#9467bd', '#2ca02c', '#d62728', 'orange', '#1f77b4']


###############################################################################
################################  Result analysis  ############################
###############################################################################

#===========================================#
# MDGMM clustering
#===========================================# 

nb_clus_tot = pd.DataFrame() 

for arch_size in ['', 'big_']:
    prefix = 'small' if arch_size == '' else 'big'

    for dataset in datasets: 
        # Plot the results
        nb_clus = pd.read_csv(results_path + 'find_nclusters/data/M1DGMM/' +\
                                    arch_size + dataset,\
                                    usecols = ['n_clusters_found'])['n_clusters_found']

        nb_clus = pd.DataFrame([nb_clus.to_list(), [dataset[6:-4]] * nb_trials]).T
        nb_clus['arch_size'] = prefix
        nb_clus.columns = ['Number of clusters found', 'design', 'arch_size']
        nb_clus_tot = nb_clus_tot.append(nb_clus, ignore_index = True)
    

ax = sns.boxplot(x = "design", y = 'Number of clusters found', hue = 'arch_size',
             data=nb_clus_tot, palette="Set3")
    
ax.tick_params('x', rotation = 30)
ax.set_ylabel('Number of clusters automatically identified') 
ax.set_title('Number of clusters identified over 30 runs for each data design')
ax.axhline(n_clusters, label = 'True number of classes', color = 'orange', linestyle = 'dashed')

# Save the figure
plt.tight_layout()
plt.rcParams["figure.figsize"] = (10,20)
plt.savefig(results_path + 'find_nclusters/figures/M1DGMM.png')
plt.show()
        
            

#===========================================#
# Hierarchical clustering
#===========================================# 

hierarch_res = pd.read_csv(results_path + 'find_nclusters/data/Hierarchical/hierarchical.csv')
hierarch_res.columns = ['design', 'linkage', 'dist_threshold', 'Number of clusters found']
hierarch_res['design'] = [d[6:-4] for d in hierarch_res.design.to_list()] # Dirty..

hierarch_res['Number of clusters found'] =  hierarch_res['Number of clusters found'].astype(int)

linkages = ['complete', 'average', 'single']
    

fig, axs = plt.subplots(2,4, figsize = (20, 10))
for d_idx, design in enumerate(designs): 
    # Plot the results
    data = hierarch_res[hierarch_res['design'] == design]
    
    for idx, linky in enumerate(linkages): 
        axs[d_idx  // 4, d_idx % 4].plot(data[data['linkage'] == linky].set_index('dist_threshold')[['Number of clusters found']],\
                 color = colors[idx], label = linky)
            
    if d_idx == 3:
        axs[d_idx  // 4, d_idx % 4].legend(title = 'initialisation', markerscale=2., loc = 'upper right')
    else:
        axs[d_idx  // 4, d_idx % 4].legend().set_visible(False)
        
    axs[d_idx  // 4, d_idx % 4].axhline(n_clusters, label = 'True number of classes', color = 'orange', linestyle = 'dashed')    
    axs[d_idx  // 4, d_idx % 4].set_yscale('log')
    axs[d_idx  // 4, d_idx % 4].set_ylabel('Clusters found')
    axs[d_idx  // 4, d_idx % 4].set_xlabel('Distance threshold')
    
    axs[d_idx  // 4, d_idx % 4].set_title(design)
    
plt.tight_layout()
plt.rcParams["figure.figsize"] = (20,10)
plt.savefig(results_path + 'find_nclusters/figures/hierarchical.png')
plt.show()
        
    
    
    
#===========================================#
# DBSCAN clustering
#===========================================# 

# Invariant to leaf size, gower fails,
# A rechecker

dbs_res = pd.read_csv(results_path + 'find_nclusters/data/DBSCAN/dbscan.csv')
dbs_res['n_clusters_found'] =  dbs_res['n_clusters_found'].astype(int)

dbs_res.columns = ['design'] + dbs_res.columns[1:-1].to_list() + ['Number of clusters found']
dbs_res['design'] = [d[6:-4] for d in dbs_res.design.to_list()] # Dirty..

lf_size = np.arange(1,6) * 10
epss = np.linspace(0.01, 5, 5)
min_ss = np.arange(1, 5)
data_to_fit = ['scaled', 'gower']
    
  

fig, axs = plt.subplots(2, 4, figsize = (20, 10))
for d_idx, design in enumerate(designs): 
    # Plot the results
    
    # Keep only interesting data
    data = dbs_res[dbs_res['design'] == design]
    #data = data[data['min_samples'] >= 2]
    data = data[data['data'] == 'scaled']
    
    for idx, eps in enumerate(epss): 
        axs[d_idx  // 4, d_idx % 4].scatter(data[data['eps'] == eps]['min_samples'],\
                                            data[data['eps'] == eps].set_index('min_samples')[['Number of clusters found']],\
                 color = colors[idx], label = eps)
            
    if d_idx == len(designs) - 1:
        axs[d_idx  // 4, d_idx % 4].legend(title = 'eps', markerscale=2.)
    else:
        axs[d_idx  // 4, d_idx % 4].legend().set_visible(False)
            
    axs[d_idx  // 4, d_idx % 4].axhline(n_clusters, label = 'True number of classes',\
                                        color = 'orange', linestyle = 'dashed')    
    axs[d_idx  // 4, d_idx % 4].set_ylabel('Clusters found')
    axs[d_idx  // 4, d_idx % 4].set_xlabel('Min samples')
    
    axs[d_idx  // 4, d_idx % 4].set_title(design)
    
plt.tight_layout()
#plt.savefig(results_path + 'find_nclusters/figures/DBSCAN.png')
plt.show()
        

data.groupby(['min_samples', 'data', 'eps', 'leaf_size']).mean().to_csv('resres.csv')
