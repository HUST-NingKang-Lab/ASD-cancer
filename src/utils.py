import pandas as pd
import numpy as np
import re
import os
from lifelines import KaplanMeierFitter 
from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc

def stage_normalize(stage):
    stage = str(stage)
    if re.search('Stage IV', stage):
        return 'Stage IV'
    elif re.search('Stage III', stage):
        return 'Stage III'
    elif re.search('Stage II', stage):
        return 'Stage II'
    elif re.search('Stage I', stage):
        return 'Stage I'
    else:
        return 'Not Available'
    
def draw_lifelines(data, cluster_col, title, save_path):
    plt.cla()
    plt.figure(figsize = (5, 5), dpi = 300)
    
    logrank = multivariate_logrank_test(data['OS.time'], data[cluster_col], data['OS'])
    ax = plt.subplot()
    
    kmf_list = []
    
    for cluster, group in data.groupby(cluster_col):  
        kmf = KaplanMeierFitter()      
        kmf.fit(group['OS.time'], group['OS'], label = f'group{cluster} (n = {group.shape[0]})')
        ax = kmf.plot_survival_function(ax = ax)
        kmf_list.append(kmf)
       
    plt.annotate(f'p-value: {logrank.p_value:.2e}', xy = (0.5, 0.5), xycoords = 'axes fraction')
    plt.title(f'{title}', fontsize = 15)
    plt.ylabel('Survival probability')
    plt.savefig(save_path, dpi = 300)
    plt.close()

class RankNorm:
    def __init__(self):
        pass
    
    def fit_transform(self, data):
        data = rankdata(data, axis = 1)/data.shape[1]
        
        return data
    
def CoxPH(X, y):
    cph = CoxPHFitter(penalizer = 0.1)
    cph_df = pd.DataFrame(np.concatenate((X, y), axis = 1))
    pval = pd.DataFrame(columns = ['p'])
    
    for i in range(cph_df.shape[1] - 2):    # -2 for OS and OS.time
        cph.fit(cph_df.loc[:, [i, cph_df.shape[1] - 2, cph_df.shape[1] - 1]],
                duration_col = cph_df.shape[1] - 2, 
                event_col = cph_df.shape[1] - 1)
        pval = pd.concat([pval, pd.DataFrame({'p': cph.summary['p']})], axis = 0)
    
    return pval

def RF_feature_importance(X, y):
    feature_name = X.columns
    X = X.values
    y = y.values - 1
    
    clf = RandomForestClassifier(n_jobs = -1)
    clf.fit(X, y)
        
    return pd.DataFrame({'feature': feature_name, 'importance': clf.feature_importances_}).sort_values('importance', ascending = False)

def predict_subtypes(X, y):
    X = X.values
    y = y.values - 1    # -1 convert group to 0, 1 labels
    
    loo = LeaveOneOut()
    y_score = np.zeros((X.shape[0]))
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf = RandomForestClassifier(n_jobs = -1)
        clf.fit(X_train, y_train)
        y_score[test_index] = clf.predict_proba(X_test)[:, 1]
        clf.fit(X_train, y_train)
        y_score[test_index] = clf.predict_proba(X_test)[:, 1]
         
    fpr, tpr, _ = roc_curve(y, y_score) 
    auroc = auc(fpr, tpr)
    
    os.makedirs
          
    return auroc
        
        
        