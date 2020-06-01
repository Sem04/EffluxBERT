# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import math
from sklearn import svm
from sklearn import neighbors
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import os

import os
os.chdir("D:/LAB PROJECT/JOURNALS/15 - Efflux v2/CODE/")
from plot_functions import *

# =============================================================================
# SET PATH AND WORKING DIRECTORY
# =============================================================================
# Set working directory
WORK_DIR = "D:/LAB PROJECT/JOURNALS/15 - Efflux v2/DATA/5 - BERT Features/"
WORK_DIR = WORK_DIR+"cased_L-12_H-768_A-12/"

# =============================================================================
# READ DATA
# =============================================================================
# Read testing data
read_test_class = pd.read_csv(WORK_DIR+'efflux_transport_test.csv')

# Read training data
read_train_class = pd.read_csv(WORK_DIR+'efflux_transport_train.csv')

# Print
# Make sure all classes for each dataset in the first column
read_train_class.head(5)


# Data and label/class
# BERT-Base, Uncased: 12-layer, 768-hidden (768 * 20 = 15360 features)
# BERT-Large, Uncased: 24-layer, 1024-hidden (1024 * 20 = 20480 features)

# Train
train_x, train_y = read_train_class.iloc[:,1:].values, read_train_class.iloc[:,0].values.tolist()

# Test
test_x, test_y = read_test_class.iloc[:,1:].values, read_test_class.iloc[:,0].values.tolist()


# =============================================================================
# TRADITIONAL MACHINE LEARNING ALGORITHMS
# =============================================================================
print("Training Gaussian Naive Bayes classifier:");
my_classifier = GaussianNB(priors=None)
my_classifier.fit(train_x, train_y)
pred_lbl_GNB = my_classifier.predict(test_x) # Prediction label/class
pred_prb_GNB = my_classifier.predict_proba(test_x); # predict probability for all target labels

print(" Training Random Forest classifier:");
my_classifier = RandomForestClassifier(max_depth=10, n_estimators=30)
my_classifier.fit(train_x, train_y)
pred_lbl_RFC = my_classifier.predict(test_x) # Prediction label/class
pred_prb_RFC = my_classifier.predict_proba(test_x); # predict probability for all target labels

print(" Training Nearest Neighbors classifier:");
n_neighbors = 100; # Optional (default = 5)
weights = 'uniform' # str or callable, optional (default = 'uniform'), 'distance'
algorithm = 'kd_tree' # {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
leaf_size = 50; # Optional (default = 30)
my_classifier = neighbors.KNeighborsClassifier()
my_classifier.fit(train_x, train_y)
pred_lbl_KNN = my_classifier.predict(test_x) # Prediction label/class
pred_prb_KNN = my_classifier.predict_proba(test_x); # predict probability for all target labels

print("Training Support Vector Machine - SVM - classifier: rbf");
kernel ='rbf'; # Optional (default='rbf') 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. If none is given, 'rbf' will be used.
C = [1]  # SVM regularization parameter, optional (default=1.0)
gamma = [0.01] # float, optional (default='auto')
for cost_ in C:
    for gamma_val in gamma:
        my_classifier = svm.SVC(kernel=kernel, gamma=gamma_val, C=cost_, probability=True)
        my_classifier.fit(train_x, train_y)
        pred_lbl_SVM = my_classifier.predict(test_x) # Prediction label/class
        pred_prb_SVM = my_classifier.predict_proba(test_x); # predict probability for all target labels


# =============================================================================
# PLOT ROC RUCVE
# =============================================================================
print('Plot ROC curve -- For all classifiers');
# Get all data first from objs
test_classes = test_y;
pred_prob_GNB = pred_prb_GNB[:,1]; # Pos label in index position 1
pred_prob_RFC = pred_prb_RFC[:,1];
pred_prob_KNN = pred_prb_KNN[:,1];
pred_prob_SVM = pred_prb_SVM[:,1];

# Output file and plot
output_plot_file = "./figures/all_classifiers_ROC_CURVE_efflux_transport.png";
set_title = "ROC Curve Analysis [Efflux vs Transport]";
plotROC_all_classifiers(set_title, test_classes, pred_prob_SVM, pred_prob_RFC, pred_prob_GNB, pred_prob_KNN, output_plot_file) # Function


print("\nTHANK YOU :)\n\n");






















