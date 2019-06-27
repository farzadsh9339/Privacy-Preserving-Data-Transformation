# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:52:38 2019

@author: farzad
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:18:10 2019

@author: farzad
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import random_projection

#Import Data
data = pd.read_csv('data.csv', header=0)


LE = LabelEncoder()
data['code'] = LE.fit_transform(data['Activity'])
y_utility = np.array(data['code'])
X = np.array(data.drop(['Activity', 'subject', 'code'],1))
# Normalization
#X_train = preprocessing.scale(X_train)
#X_train = pd.DataFrame(X_train)

y_privacy = np.array(data['subject'])


# Random Projection Transformation from 1 to 30 features
Accuracy_U_RP = []
Accuracy_P_RP = []
for i in range (1, 5):    
    transformer = random_projection.GaussianRandomProjection(n_components=i)
    X_transformed = transformer.fit_transform(X)
    X_train_transformed, X_test_transformed, y_train_utility, y_test_utility = train_test_split(X_transformed, y_utility, test_size=0.20, random_state=42)
#    X_test_transformed = pca.fit_transform(X_test) 
    
#UTILITY TARGET CLASSIFICATION SVM USING DIFFERENT KERNELS WITH 5 FOLD CROSS-VALIDATION USING DEFFERENT GAMMAS AND C's 

    parameter_candidates = {'kernel': ('rbf','poly'), 'C':[1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1, cv=5)
    clf.fit(X_train_transformed,y_train_utility)
    
    accuracy_utility_train = clf.best_score_

    accuracy_utility = svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel, gamma=clf.best_estimator_.gamma).fit(X_train_transformed, y_train_utility).score(X_test_transformed, y_test_utility)
    Accuracy_U_RP.append(accuracy_utility)
    
    
    
#PRIVACY TARGET CLASIFICATION SVM USING DIFFERENT KERNELS WITH 5 FOLD CROSS-VALIDATION USING DEFFERENT GAMMAS AND C's 
    
    X_train_transformed, X_test_transformed, y_train_privacy, y_test_privacy = train_test_split(X_transformed, y_privacy, test_size=0.20, random_state=42)
    parameter_candidates = {'kernel': ('rbf','poly'), 'C':[1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1, cv=5)
    clf.fit(X_train_transformed,y_train_privacy)
    
    accuracy_privacy_train = clf.best_score_

    accuracy_utility = svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel, gamma=clf.best_estimator_.gamma).fit(X_train_transformed, y_train_privacy).score(X_test_transformed, y_test_privacy)
    Accuracy_P_RP.append(accuracy_utility)
    
