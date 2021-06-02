# -*- coding: utf-8 -*-
"""
Created on Mon April 29 13:25:11 2020

@author: rfuchs
"""


import os 

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
from table_evaluator import TableEvaluator

###############################################################################
######################         Adult data          ############################
###############################################################################


#===========================================#
# Model Hyper-parameters
#===========================================#

n_clusters = 'auto'
r = np.array([3, 1])
k = [4]

seed = 1
init_seed = 2
    
# !!! Changed eps
eps = 1E-01
it = 10
maxstep = 100
        
var_distrib = np.array(['continuous', 'categorical', 'continuous', 'ordinal',\
                        'binomial', 'categorical', 'categorical', 'categorical',\
                        'categorical', 'bernoulli', 'ordinal', 'ordinal',\
                        'continuous', 'categorical', 'bernoulli']) 

# Plotting utilities
varnames = ['age', 'workclass', 'fnlwgt', 'education', 
            'education.num', 'marital.status', 'occupation', 'relationship',\
            'race', 'sex', 'capital.gain', 'capital.loss',\
            'hours.per.week', 'native.country', 'income']
    
p_new = len(var_distrib)

#===========================================#
# Importing data
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')

experiment_designs = ['Absent', 'Small', 'Unbalanced']
nb_files_per_design = 10
inf_nb = 1E12


'''
design = experiment_designs[0]
filenum = 1
'''

acceptance_rate = dict(zip(experiment_designs, [[],[],[]]))

for design in experiment_designs:
    
    #*****************************************************************
    # File name formatting and sampling rules
    #*****************************************************************    
    
    if design in ['Absent', 'Unbalanced']:
        prefix = design[:3] + '_'
        # Want to sample only women of more than 60 years old
        authorized_ranges = np.expand_dims(np.stack([[-inf_nb,inf_nb] for var in var_distrib]).T, 1)
        authorized_ranges[:,0, 0] = [60, 100]  # Of more than 60 years old
        authorized_ranges[:,0, 9] = [0, 0] # Only women
        nb_pobs = 200

    elif design == 'Small':
        prefix = '' # 
        authorized_ranges = np.expand_dims(np.stack([[-inf_nb,inf_nb] for var in var_distrib]).T, 1)
        nb_pobs = 5000


    #*****************************************************************
    # Generate data for all experiment design
    #*****************************************************************       
    
    for filenum in range(1, nb_files_per_design + 1):
        if design in ['Absent', 'Unbalanced']:
            train_filepath = 'adult/' + design + '/' + prefix + 'Train' + str(filenum) + '.csv'
        elif design == 'Small':
            train_filepath = 'adult/' + design + '/' + prefix + 'Train_' + str(filenum) + '.csv'
        else:
            raise RuntimeError('Please specify of valid design')
     
        train = pd.read_csv(train_filepath, sep = ',')
        
        train = train.infer_objects()
        p = train.shape[1]
        
        # Import the test set
        test_filepath = 'adult/' + design + '/' + prefix + 'Test_' + str(filenum) + '.csv'
        test = pd.read_csv(test_filepath, sep = ',')
        
        # Delete the missing values 
        train = train.loc[~(train == '?').any(1)]
        test = test.loc[~(test == '?').any(1)]
        numobs = len(train)
        
        #*****************************************************************
        # Formating the data
        #*****************************************************************
                       
        # Encode categorical datas
        categ_dict = {} # Store the Label encoding
        for col_idx, colname in enumerate(train.columns):
            if var_distrib[col_idx] == 'categorical': 
                le = LabelEncoder()

                # Keep only the modalities that the two datasets have in common
                test = test[test[colname].isin(list(set(train[colname])))]

                # Convert them into numerical values               
                train[colname] = le.fit_transform(train[colname]) 
                categ_dict[colname] = deepcopy(le)

            
        # Encode binary data
        bin_dict = {} # Store the observations
        for col_idx, colname in enumerate(train.columns):
            le = LabelEncoder()

            if var_distrib[col_idx] == 'bernoulli': 
                train[colname] = le.fit_transform(train[colname])
                bin_dict[colname] = deepcopy(le)

        # Encode ordinal data, modalities have been sorted (at best)
        ord_modalities = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th',\
                          '10th', '11th', '12th', 'HS-grad', 'Some-college','Masters',\
                          'Bachelors', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Doctorate']
            
        # Delete non-existing modalities in the test set 
        test = test[test['education'].isin(ord_modalities)]
        
        for idx, mod in enumerate(ord_modalities):
            train['education'] = np.where(train['education'] == mod, idx, train['education'])
                    
        train['education'] = train['education'].astype(int)
        ord_le = LabelEncoder()
        train['education.num'] = ord_le.fit_transform(train['education.num'])

        # Encode capital.gain and capital.loss and capital.gain as ordinal variables
        k_dict = {}
        nb_bins = 5 # The size of each class
        for col in ['capital.gain', 'capital.loss']:
            le = LabelEncoder()

            step = np.ceil(test[col].max()) / nb_bins
            
            # Create the intervals for each class
            bins = pd.IntervalIndex.from_tuples([(-0.5, 0.5)] +\
                                [(1 + i*step, 1 + (i + 1) * step) for i in range(nb_bins)])
                
            discrete_k = pd.cut(train[col], bins).map(lambda x: x.mid).astype(float)
            print(set(discrete_k))
            test[col] = pd.cut(test[col], bins).map(lambda x: x.mid).astype(float)
            
            le.fit(test[col])
            train[col] = le.transform(discrete_k)
            k_dict[col] = deepcopy(le)

        nj, nj_bin, nj_ord, nj_categ = compute_nj(train, var_distrib)
        nb_cont = np.sum(var_distrib == 'continuous')        
        p_new = train.shape[1]
        train_np = train.values
                
        # Defining distances over the features
        cat_features = pd.Series(var_distrib).isin(['categorical', 'bernoulli']).to_list()
        dm = gower_matrix(train.astype(np.object), cat_features = cat_features) 
        
        dtype = {train.columns[j]: np.float64 if (var_distrib[j] != 'bernoulli') and \
                (var_distrib[j] != 'categorical') else np.str for j in range(p_new)}
        
        train = train.astype(dtype, copy=True)
        numobs = len(train)
        
        #*****************************************************************
        # Run MIAMI
        #*****************************************************************
                                  
        prince_init = dim_reduce_init(train, 2, k, r, nj, var_distrib, seed = None,\
                                      use_famd=True)
        out = MIAMI(train_np, 'auto', r, k, prince_init, var_distrib, nj, authorized_ranges, nb_pobs, it,\
                     eps, maxstep, seed, perform_selec = False, dm = dm, max_patience = 0)
        
        print('MIAMI has kept one observation over', round(1 / out['share_kept_pseudo_obs']),\
              'observations generated')
        acceptance_rate[design].append(out['share_kept_pseudo_obs'])
        
        #*****************************************************************
        # Visualisation result
        #*****************************************************************
        
        train_new_np = out['y_all'][len(train):]
        train_new = pd.DataFrame(train_new_np, columns = train.columns)
 
        le_dict = {**categ_dict, **bin_dict, **k_dict} 
        le_dict['education.num'] = ord_le

        # Relabel the data 
        for col_idx, colname in enumerate(train.columns):
            if (var_distrib[col_idx] != 'continuous') & (colname != 'education'): 
                le = le_dict[colname]
                train_new[colname] = le.inverse_transform(train_new[colname].astype(int)) 
        
        for idx, mod in enumerate(ord_modalities):
            train_new['education'] = np.where(train_new['education'] == str(float(idx)), mod, train_new['education'])
                
        # Keep only the women that have more than 60 years in the test
        if design in ['Absent', 'Unb']:
            test = test[(test['age'] >= 60) & (test['sex'] == 'Female')]
        
        # Store the predictions
        train_new.to_csv('pseudo_adult/' + design + '/' + 'preds' + str(filenum) + '.csv',\
                         index = False)
        
            
# Visualise the predictions
# Use table_evaluator    
plt.plot([0])
plt.title('Design:' + design)
plt.show()
cat_features = (~(pd.Series(var_distrib) != 'continuous')).to_list()
table_evaluator = TableEvaluator(test, train_new)
table_evaluator.visual_evaluation()


