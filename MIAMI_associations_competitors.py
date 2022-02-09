# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 15:33:43 2022

@author: rfuchs
"""

import os 

import re 
os.chdir('C:/Users/rfuchs/Documents/GitHub/M1DGMM')

import pandas as pd
from copy import deepcopy
from gower import gower_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 

from miami import MIAMI
from init_params import dim_reduce_init
from data_preprocessing import compute_nj

import autograd.numpy as np
from dython.nominal import associations
#from table_evaluator import TableEvaluator

from pandas.errors import ParserError
#import seaborn as sns
import seaborn as sns

###############################################################################
######################         Adult data          ############################
###############################################################################

data_folder = 'C:/Users/rfuchs/Documents/These/Stats/MIAMI/Datasets/Adult/'
res_folder = 'C:/Users/rfuchs/Documents/These/Stats/MIAMI/Results/Adult data/'


var_distrib = np.array(['continuous', 'categorical', 'continuous',\
                        'ordinal', 'categorical', 'categorical', 'categorical',\
                        'categorical', 'bernoulli', 'ordinal', 'ordinal',\
                        'continuous', 'categorical', 'bernoulli']) 

# Plotting utilities
varnames = np.array(['age', 'workclass', 'fnlwgt',\
            'education.num', 'marital.status', 'occupation', 'relationship',\
            'race', 'sex', 'capital.gain', 'capital.loss',\
            'hours.per.week', 'native.country', 'income'])
    
p = len(varnames)

dtypes_dict = {'continuous': float, 'categorical': str, 'ordinal': float,\
              'bernoulli': str, 'binomial': int}

cat_features = np.logical_or(var_distrib == 'categorical', var_distrib == 'bernoulli')

#===================================== 
# Select the design
#===================================== 

design = 'Absent'
filenum = 1
sub_design = 'trivarié'
prefix = design[:3] + '_'
nb_files_per_design = 10
nb_pobs = 200
sub_aliases = {'bivarié': 'bivariate','trivarié': 'trivariate'}


#===================================== 
# Import the train and test sets
#===================================== 

train = pd.read_csv(data_folder + design + '/'  + sub_design + '/' + prefix + 'Train_' +\
                       str(filenum) + '.csv', sep = ';')
            
test = pd.read_csv(data_folder + design + '/'  + sub_design + '/' + prefix + 'Test_F_' +\
                       str(filenum) + '.csv', sep = ';')

del(train['education'])
del(test['education'])

dtype = {train.columns[j]: dtypes_dict[var_distrib[j]] for j in range(p)}

# Keep the observations that present the desired modalities
if sub_design == 'bivarié':
    test = test[(test['age'] > 60) & (test['sex'] == 'Female')]
elif sub_design == 'trivarié':
    test = test[(test['age'] > 60) & (test['sex'] == 'Female') &\
                (test['marital.status'] == 'Widowed')]
    
# Keep only the modalities existing in the train 
for j, col in enumerate(train.columns):
    if not(var_distrib[j] in ['continuous', 'binomial']):
        train_modalities = list(set(train[col]))
        test = test[test[col].isin(train_modalities)]
        
test['capital.loss'] = [float(re.sub(',', '.', aa)) for aa in  test['capital.loss']]
test['capital.gain'] = [float(re.sub(',', '.', aa)) for aa in  test['capital.gain']]

train['capital.loss'] = [float(re.sub(',', '.', aa)) for aa in  train['capital.loss']]
train['capital.gain'] = [float(re.sub(',', '.', aa)) for aa in  train['capital.gain']]

#===================================== 
# Select the test set
#===================================== 
competitors = os.listdir(res_folder + design + '/' + sub_design)


assocs = {}
assoc_test = associations(test.astype(dtype), nominal_columns = list(test.columns[cat_features]),\
                          plot = False)['corr']
   
#ax = axs[0], cbar = False, annot = False
assocs['Test'] = assoc_test

for c_idx, competitor in enumerate(competitors):
    try:  
        preds = pd.read_csv(res_folder + design + '/' + sub_design +  '/' + competitor\
                           + '/preds' + str(filenum) + '.csv', sep = ',')
        if preds.shape[1] == 1:
            preds = pd.read_csv(res_folder + design + '/' + sub_design +  '/' + competitor\
                               + '/preds' + str(filenum) + '.csv', sep = ';')
        #if preds.shape[1] == p + 1:
            #preds = preds.iloc[:, 1:]
    except ParserError:
        preds = pd.read_csv(res_folder + design + '/' + sub_design +  '/' + competitor\
                           + '/preds' + str(filenum) + '.csv', sep = ';')
    except FileNotFoundError:
        print(competitor + ' has no results')
        continue
            
    if 'Y' in preds.columns:
        del(preds['Y'])
        
    if preds.shape[1] == p + 1:
        del(preds['education'])
        
    assert preds.shape[0] == 200
    assert preds.shape[1] == p
    preds = preds[test.columns]
    #assert (preds.columns == test.columns).all()
    
    preds['capital.loss'] = [float(re.sub(',', '.', str(aa))) for aa in  preds['capital.loss']]
    preds['capital.gain'] = [float(re.sub(',', '.', str(aa))) for aa in  preds['capital.gain']]

    
    assoc_preds = associations(preds.astype(dtype), nominal_columns = list(preds.columns[cat_features]),\
                               plot = False, num_num_assoc = 'kendall')['corr']
    
    assocs[competitor] = assoc_preds
    

for c_idx, competitor in enumerate(assocs.keys()):
    sns.heatmap(assocs[competitor], vmin = 0, annot = False)
    plt.title(competitor)
    plt.tight_layout()
    #plt.savefig('C:/Users/rfuchs/Documents/These/Stats/MIAMI/plots/Adult/' +\
                #design + '/' + sub_design + '/' + competitor + '.png')
    plt.show()





