# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:42:30 2019

@author: farzad
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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


# Calculation of scatter matrices for both utility and privacy labels(scatter between-class & within class)

def comp_mean_vectors(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(X[y==cl], axis=0))
    return mean_vectors

def scatter_between(X, y):
    overall_mean = np.mean(X, axis=0)
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)    
    S_Bu = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):  
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(n_features, 1) 
        overall_mean = overall_mean.reshape(n_features, 1) 
        S_Bu += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return S_Bu
    
def scatter_within(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    S_W = np.zeros((n_features, n_features))
    for cl, mv in zip(class_labels, mean_vectors):
        class_sc_mat = np.zeros((n_features, n_features))                 
        for row in X[y == cl]:
            row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1) 
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat                           
    return S_W


# add regularization 
S_Bu, S_Bp = scatter_between(X, y_utility), scatter_between(X, y_privacy)
Regularization = np.identity(X.shape[1])*0.0001
S_Bu_reg = S_Bu + Regularization
S_Bp_reg = S_Bp + Regularization


eigen_vals, eigen_vecs = np.linalg.eigh(np.linalg.inv(S_Bp_reg).dot(S_Bu_reg))


# getting the eigen value and eigen vectors for W matrix
def get_components(eigen_vals, eigen_vecs, n_comp):
   # n_features = Xu.shape[1]
    eig_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.hstack([eig_pairs[i][1].reshape(X.shape[1], 1) for i in range(0, n_comp)])
    return W




# MDR Transformation from 1 to 15 features
Accuracy_U_MDR = []
Accuracy_P_MDR = []
for i in range (1, 15):    
    X_transformed = X.dot(get_components(eigen_vals, eigen_vecs, n_comp=i))
    X_train_transformed, X_test_transformed, y_train_utility, y_test_utility = train_test_split(X_transformed, y_utility, test_size=0.20, random_state=42)    
    
#UTILITY TARGET CLASSIFICATION SVM USING DIFFERENT KERNELS WITH 5 FOLD CROSS-VALIDATION USING DEFFERENT GAMMAS AND C's 

    parameter_candidates = {'kernel': ('rbf','poly'), 'C':[1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1, cv=5)
    clf.fit(X_train_transformed,y_train_utility)
    
    accuracy_utility_train = clf.best_score_

    accuracy_utility = svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel, gamma=clf.best_estimator_.gamma).fit(X_train_transformed, y_train_utility).score(X_test_transformed, y_test_utility)
    Accuracy_U_MDR.append(accuracy_utility)
    
    
    
#PRIVACY TARGET CLASIFICATION SVM USING DIFFERENT KERNELS WITH 5 FOLD CROSS-VALIDATION USING DEFFERENT GAMMAS AND C's 
    X_train_transformed, X_test_transformed, y_train_privacy, y_test_privacy = train_test_split(X_transformed, y_privacy, test_size=0.20, random_state=42)
    parameter_candidates = {'kernel': ('rbf','poly'), 'C':[1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1, cv=5)
    clf.fit(X_train_transformed,y_train_privacy)
    
    accuracy_privacy_train = clf.best_score_

    accuracy_privacy = svm.SVC(C=clf.best_estimator_.C, kernel=clf.best_estimator_.kernel, gamma=clf.best_estimator_.gamma).fit(X_train_transformed, y_train_privacy).score(X_test_transformed, y_test_privacy)
    Accuracy_P_MDR.append(accuracy_privacy)