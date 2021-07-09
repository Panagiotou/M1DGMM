# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:15:52 2021

@author: rfuchs
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 

from dython.nominal import compute_associations
from table_evaluator import TableEvaluator
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score



os.chdir(r'C:\Users\rfuchs\Documents\These\Stats\mixed_dgmm\datasets')

train_test_folder = 'adult'
fake_folder = 'Adult data'

varnames = ['age', 'workclass', 'fnlwgt', 'education', 
            'education.num', 'marital.status', 'occupation', 'relationship',\
            'race', 'sex', 'capital.gain', 'capital.loss',\
            'hours.per.week', 'native.country', 'income']

var_distrib = np.array(['continuous', 'categorical', 'continuous', 'ordinal',\
                        'binomial', 'categorical', 'categorical', 'categorical',\
                        'categorical', 'bernoulli', 'ordinal', 'ordinal',\
                        'continuous', 'categorical', 'bernoulli']) 
p = len(var_distrib)
cat_features = (pd.Series(var_distrib) != 'continuous').to_list()
dtype = {varnames[j]: np.str if cat_features[j] 
         else  np.float64  for j in range(p)}




#====================================
# Make some 3D plots with matplotlib
#====================================

designs = ['Absent', 'Small', 'Unbalanced']
datasets = ['preds' + str(i) + '.csv' for i in range(1,11)]

x = 'capital.loss' 
y = 'fnlwgt'
z = 'hours.per.week'

f1_scores_dict = {}

for design in designs:
    models = os.listdir(fake_folder + '/' + design)
    nb_models = len(models)

    data_prefix = '' if design == 'Small' else design[:3] + '_' 

    nb_plots = 3 if design != 'Absent' else 2
    offset = 1 if design != 'Absent' else 0 # To take into account the lack of train data for Absent
    
    f1_scores = pd.DataFrame(columns = ['set_idx', 'classifier', 'model', 'f1'])

    for idx in range(1,11):
        
        train = pd.read_csv(train_test_folder + '/' + design + '/' +\
                                 '/' + data_prefix + 'Train' +  str(idx) + '.csv')
        test = pd.read_csv(train_test_folder + '/' + design + '/' +\
                                 '/' + data_prefix + 'Test_' +  str(idx) + '.csv')
        
        #==========================================
        # Trivariate distributions
        #==========================================
        
        
        # Keep only the interesting modalities  
        if design in ['Absent', 'Unbalanced']:
            train = train[(train['age'] >= 60) & (train['sex'] == 'Female')]
            test = test[(test['age'] >= 60) & (test['sex'] == 'Female')]
         
        data = {}
        for model_idx, model in enumerate(models):
            try:
                fake = pd.read_csv(fake_folder + '/' + design + '/' +  model +\
                                    '/preds' + str(idx) + '.csv')
                data[model] = fake

            except FileNotFoundError:
                print(model, ' has no preds ', str(idx))
                fake = pd.DataFrame(columns = train.columns)
                
        data['Train'] = train
        data['Test'] = test

        
        plot_datasets = list(set(data.keys())) 
        plot_datasets.sort()
        plot_datasets = [d for d in plot_datasets\
                                     if (d != 'Train') & (d != 'Test')] + ['Test']
            
        if design != 'Absent':
            plot_datasets = ['Train'] + plot_datasets
            
        fig = plt.figure(figsize = plt.figaspect(1 / len(plot_datasets)))
        #fig.suptitle(design + ' ' + ' (preds' + str(idx) + ')',\
                     #fontsize=16)

        for i, dataset in enumerate(plot_datasets):
            if (design == 'Absent') & (dataset == "Train"):
                continue
            
            ax = fig.add_subplot(1, len(plot_datasets), i + 1, projection='3d')
            ax.scatter(data[dataset][x], data[dataset][y], data[dataset][z])
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
            ax.set_xlim(data['Test'][x].min() - 10,data['Test'][x].max() + 10)    
            ax.set_ylim(data['Test'][y].min() - 10,data['Test'][y].max() + 10)
            ax.set_zlim(data['Test'][z].min() - 10,data['Test'][z].max() + 10)

            ax.set_title(dataset, fontsize = 24)
        
        #fig.savefig('../Figures/adult/' + design + '/preds' + str(idx) + '.png')
        plt.show()
        
        # Dirty hack to remove
        if design == 'Unbalanced':
            fig = plt.figure(figsize = plt.figaspect(1 / 30))

            for i, dataset in enumerate(plot_datasets[:5]):
                ax = fig.add_subplot(1, len(plot_datasets), i + 1, projection='3d')
                ax.scatter(data[dataset][x], data[dataset][y], data[dataset][z])
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_zlabel(z)
                ax.set_xlim(data['Test'][x].min() - 10,data['Test'][x].max() + 10)    
                ax.set_ylim(data['Test'][y].min() - 10,data['Test'][y].max() + 10)
                ax.set_zlim(data['Test'][z].min() - 10,data['Test'][z].max() + 10)
    
    
                if i == 4:
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_zlabel(z)

                ax.set_title(dataset, fontsize = 14) 
            plt.tight_layout()
            plt.show()

            fig = plt.figure(figsize = plt.figaspect(1 / 5))
              
            for i, dataset in enumerate(plot_datasets[5:]):

                ax = fig.add_subplot(1, len(plot_datasets), i + 1, projection='3d')
                ax.scatter(data[dataset][x], data[dataset][y], data[dataset][z])

                ax.set_xlim(data['Test'][x].min() - 10,data['Test'][x].max() + 10)    
                ax.set_ylim(data['Test'][y].min() - 10,data['Test'][y].max() + 10)
                ax.set_zlim(data['Test'][z].min() - 10,data['Test'][z].max() + 10)
                
                if i == 4:
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_zlabel(z)

    
                ax.set_title(dataset, fontsize = 14) 
            plt.tight_layout()

            plt.show()
            
        dataset = 'MIAMI'
        cols= ['age', 'workclass', 'marital.status']     
        table_evaluator = TableEvaluator(data['Test'][cols], data[dataset][cols])
    
        table_evaluator.plot_distributions()


        
        #==========================================
        # Correlation matrices
        #==========================================
        
        w_ratios = [1] * len(plot_datasets) + [0.08]
        fig, axs = plt.subplots(1, len(plot_datasets) + 1, \
                                gridspec_kw={'width_ratios':w_ratios}, figsize = (15,10))
            
        axs[0].get_shared_y_axes().join(*axs[1:-1])
        #fig.suptitle(design + ' ' + ' (preds' + str(idx) + ')',\
                     #fontsize=16)
            
        for i, dataset in enumerate(plot_datasets):
            if (design == 'Absent') & (dataset == "Train"):
                continue    
            
            # Delete useless variables
            data[dataset] = data[dataset].iloc[:, :p] # Delete legacy columns
                        
            # Encode categorical data
            for col_idx, colname in enumerate(train.columns):
                if cat_features[col_idx]: 
                    le = LabelEncoder()
                    data[dataset][colname] = le.fit_transform(data[dataset][colname]) 
            
            
            # Delete the sex column
            data[dataset] = data[dataset][[name for name in varnames if name != 'sex']]
            cat_features_new = [cat_features[i] for i in range(len(varnames)) if varnames[i] != 'sex']

        
            if i == len(plot_datasets) - 1:
                corr = compute_associations(data[dataset].astype(int), nominal_columns = cat_features_new)

                g1 = sns.heatmap(corr,cmap="YlGnBu",ax=axs[i], cbar_ax=axs[-1])

            else:
                corr = compute_associations(data[dataset].astype(int), nominal_columns = cat_features_new)
                g1 = sns.heatmap(corr,cmap="YlGnBu",cbar=False,ax=axs[i])
                
            g1.set_ylabel('')
            g1.set_xlabel('')
            g1.set_yticks([])

            axs[i].set_title(dataset, fontsize = 25)
 
        plt.tight_layout()
        plt.show()
        
        #==========================================
        # Predictive performances
        #==========================================
        
        target = 'education.num'
        covariates = data['Test'].columns.tolist()
        covariates.remove(target)
        metric = 'weighted'
        
        X_test = data['Test'][covariates]
        y_test = data['Test'][target]
        
        for i, dataset in enumerate(plot_datasets[:-1]):
            if (design == 'Absent') & (dataset == "Train"):
                continue    
                        
            X_train = data[dataset].round(0).astype(int)[covariates]
            y_train = data[dataset].round(0).astype(int)[target]
            
            clf = RandomForestClassifier(max_depth=2, random_state=0)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            print(np.unique(pred, return_counts=True))
      
                    
            f1 = f1_score(y_test, pred, average=metric)
            f1_scores = f1_scores.append({'set_idx': idx, 'classifier': 'RF',\
                                          'model': dataset, 'f1': f1}, ignore_index = True)
                
        f1_scores_dict[design] = f1_scores



for key, item in f1_scores_dict.items():
    ax = sns.boxplot(x = "model", y = "f1",
                     data=item, palette="Set3",\
                         showfliers = False)
            
    ax.tick_params('x', rotation = 30)
    ax.set_ylabel(metric + 'precision') 
    ax.set_title('[' + key + ']' + metric + ' precision to predict ' + target + 'with a RF classifier') 
    plt.show()

                
