# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:35:27 2021

@author: rfuchs
"""

import re
import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/M1DGMM')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics.cluster import adjusted_rand_score

import autograd.numpy as np

results_path = 'C:/Users/rfuchs/Documents/These/Experiences/' # Results storage
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')
datasets = os.listdir('simulated')

nb_trials = 30
nb_aris = int((nb_trials - 1) * (nb_trials / 2)) # Number of couples of ARIS to compute
n_clusters = 4

partition_repo = 'C:/Users/rfuchs/Documents/These/Experiences/similar_partition/'
designs = ['1n500', '1n1000', '1bisn500', '1bisn1000','2n500', '2n1000', '2bisn500', '2bisn1000']


###############################################################################
################################  Result analysis  ############################
###############################################################################

#===========================================#
# MDGMM clustering
#===========================================# 

# Small: r = {3, 2}, mini r = {2,1}, big = {5, 3, 1}

aris_distrib = pd.DataFrame()

archs = ['big', 'small', 'mini']

for arch in archs:
    for dataset in datasets: 
        
        mdgmm_res = pd.read_csv(partition_repo + 'data/MDGMM/' + dataset[:-4] + '_' + arch + '.csv').values
        assert mdgmm_res.shape[1] == 30

        aris = [adjusted_rand_score(mdgmm_res[:,i], mdgmm_res[:,j]) \
                for i in range(nb_trials) for j in range(nb_trials) if i < j]
        
        aris_dataset = pd.DataFrame([aris, [dataset[6:-4]] * nb_aris, [arch] * nb_aris]).T
        aris_dataset.columns = ['ARI_ij', 'design', 'Architecture']
        aris_distrib = aris_distrib.append(aris_dataset,ignore_index = True)
        
        
ax = sns.boxplot(x = 'design', y = 'ARI_ij', hue = 'Architecture',
                 data=aris_distrib, palette="Set3", order = designs)
        
ax.tick_params('x', rotation = 30)
#ax.set_yscale("log") 
ax.set_ylabel('Adjusted Rand Index') 
#ax.set_title('ARI distribution over 30 runs for each data design')

plt.tight_layout()
#plt.savefig(results_path + 'similar_partition/figures/MDGMM_ARIs_distrib.png')
plt.show()

            
#===========================================# 
# KMODES algorithm
#===========================================# 

inits = ['Huang', 'Cao', 'random']
aris_distrib = pd.DataFrame()

for dataset in datasets: 
        
    for init in inits:    
        part_res_modes = pd.read_csv(partition_repo + 'data/KMODES/' + dataset[:-4] + '_' +\
                                     init + '.csv').iloc[:,1:].values
    
        assert part_res_modes.shape[1] == 30
    
        aris = [adjusted_rand_score(part_res_modes[:,i], part_res_modes[:,j]) \
                for i in range(nb_trials) for j in range(nb_trials) if i < j]
    
        aris_dataset = pd.DataFrame([aris, [dataset[6:-4]] * nb_aris, [init] * nb_aris ]).T
        aris_dataset.columns = ['ARI_ij', 'design', 'initialisation']
        aris_distrib = aris_distrib.append(aris_dataset,ignore_index = True)
    
        
ax = sns.boxplot(x = 'design', hue = "initialisation", y = 'ARI_ij',
                 data=aris_distrib, palette="Set3", order = designs)
  
ax.legend(loc = 'lower right')  
ax.tick_params('x', rotation = 30)
ax.set_ylabel('Adjusted Rand Index') 

# Save the figs now..
plt.tight_layout()
plt.savefig(results_path + 'similar_partition/figures/KMODES_ARIs_distrib.png')
plt.show()


#===========================================# 
# KPROTOTYPES
#===========================================# 

inits = ['Huang', 'Cao', 'random']
aris_distrib = pd.DataFrame()

for dataset in datasets: 
    
    for init in inits:    
        part_res_proto = pd.read_csv(partition_repo + 'data/KPROTOTYPES/' + dataset[:-4] + '_' +\
                                     init + '.csv').values

        assert part_res_proto.shape[1] == 30


        aris = [adjusted_rand_score(part_res_proto[:,i], part_res_proto[:,j]) \
                for i in range(nb_trials) for j in range(nb_trials) if i < j]
    
        aris_dataset = pd.DataFrame([aris, [dataset[6:-4]] * nb_aris, [init] * nb_aris ]).T
        aris_dataset.columns = ['ARI_ij', 'design', 'initialisation']
        aris_distrib = aris_distrib.append(aris_dataset,ignore_index = True)
    
    
ax = sns.boxplot(x = 'design', hue = "initialisation", y = 'ARI_ij',
                 data=aris_distrib, palette="Set3", order = designs)
        
ax.tick_params('x', rotation = 30)
ax.legend(loc = 'lower right')  
ax.set_ylabel('Adjusted Rand Index') 

# Save the figs now..
plt.tight_layout()
plt.savefig(results_path + 'similar_partition/figures/KPROTOTYPES_ARIs_distrib.png')
plt.show()


#===========================================#
# Hierarchical clustering
#===========================================# 
# Perfectly deterministic

linkages = ['complete', 'average', 'single']

for design in designs: 
    
    for linky in linkages:
        
        hierarch_res  = pd.read_csv(results_path + 'similar_partition/data/Hierarchical/' +\
                            'result' + design + '_' + linky + '.csv', header = None)
            
        assert hierarch_res.std(1).sum() == 0


#===========================================# 
# SOM
#===========================================# 
aris_distrib = pd.DataFrame()

sigmas = np.linspace(0.001, 3, 5)
lrs = np.linspace(0.0001, 0.5, 10)

for dataset in datasets: 
    for sig in sigmas:
        for lr in lrs:
            som_res = pd.read_csv(partition_repo + '/data/SOM/' + \
                                  dataset[:-4] + '_' + str(sig) + '_' + str(lr) + '.csv').values
      
            aris = [adjusted_rand_score(som_res[:,i], som_res[:,j]) \
                    for i in range(nb_trials) for j in range(nb_trials) if i < j]
        
            aris_dataset = pd.DataFrame([aris, [dataset[6:-4]] * nb_aris,\
                                         [str(sig)] * nb_aris, [str(lr)] * nb_aris ]).T
            aris_dataset.columns = ['ARI_ij', 'design', 'sigma', 'learning rate']
            aris_distrib = aris_distrib.append(aris_dataset,ignore_index = True)
            

fig, axs = plt.subplots(2, 4, figsize = (20, 10))
for d_idx, design in enumerate(designs): 
    data = aris_distrib[aris_distrib['design'] == design]
    data['sigma'] = data['sigma'].astype(float).round(4)
    data['learning rate'] = data['learning rate'].astype(float).round(3)
    
    sns.boxplot(x = 'learning rate', hue = "sigma", y = 'ARI_ij',
                 data=data, palette="Set3", ax = axs[d_idx  // 4, d_idx % 4])

    if d_idx == len(designs) - 1:
        axs[d_idx  // 4, d_idx % 4].legend(title = 'sigma', markerscale=100.)
    else:
        axs[d_idx  // 4, d_idx % 4].legend().set_visible(False)
        
                        
    axs[d_idx  // 4, d_idx % 4].set_ylabel('ARI')
    axs[d_idx  // 4, d_idx % 4].set_xlabel('learning rate')
    axs[d_idx  // 4, d_idx % 4].set_title(design)
 
#plt.legend(sigmas, title = 'sigma', markerscale = 100.)
plt.tight_layout()
plt.savefig(results_path + 'similar_partition/figures/SOM_ARIs_distrib.png')

plt.show()


        

#===========================================#
# DBSCAN clustering
#===========================================# 
lf_size = np.arange(1,6) * 10
epss = np.linspace(0.01, 5, 5)
min_ss = np.arange(1, 5)
data_to_fit = ['scaled', 'gower']
    
for design in designs: 
    
    for lfs in lf_size:
        print("Leaf size:", lfs)
        for eps in epss:
            for min_s in min_ss:
                for data in data_to_fit:
                    dbs_res= pd.read_csv(results_path + 'similar_partition/data/DBSCAN/dbscan' + \
                                                   dataset[:-4] + '_' + str(lfs) + '_' + str(eps) + '_' + str(min_s) + '_' +\
                                                       str(data) + '.csv', header = None)
                    
                    assert dbs_res.std(1).sum() == 0
       
            


