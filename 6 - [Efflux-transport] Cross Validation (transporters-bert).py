# -*- coding: utf-8 -*-

# Load Packages
import pandas as pd 
import numpy as np
import os
import math
from sklearn import svm
from sklearn import neighbors
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
seed = 7 # fix random seed for reproducibility
np.random.seed(seed)
from imblearn.over_sampling import SMOTE 

##########################################################
# Confusion matrix
##########################################################
def ConfusionMatrixBinaryClassifier(number_fold, classifier_name, test_label_original, test_label_predicted, saveFileName):
    # Term for confusion matrix
    TP=0.0;FP=0.0;FN=0.0;TN=0.0;
    
    for count in range(0, len(test_label_original)):
        predictedClass = test_label_predicted[count]
        expectedClass = test_label_original[count];

        # Basic calculation of confusion matrix 
        if predictedClass == 1 and expectedClass == 1:
            TP=TP+1;

        if predictedClass == 1 and expectedClass == 0:
            FP=FP+1;

        if predictedClass == 0 and expectedClass == 1:
            FN=FN+1;

        if predictedClass == 0 and expectedClass == 0:
            TN=TN+1;
    
    Sensitivity=0.0; Specificity=0.0;Precision=0.0;Accuracy=0.0;
    MCC=0.0;F1=0.0;
    try:
        Sensitivity=round((TP/(TP+FN)), 4);
        Specificity=round(TN/(FP+TN), 4);
        Precision=round(TP/(TP+FP), 4);
        Accuracy = round((TP+TN)/(TP+FP+TN+FN), 4);
        MCC=round(((TP*TN)-(FP*FN))/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))), 4);
        F1=round((2*TP)/((2*TP)+FP+FN), 4);
    
    except:
        Sensitivity=0.0; Specificity=0.0;Precision=0.0;Accuracy=0.0;
        MCC=0.0;F1=0.0;
    
    if TP == 44 and TN == 820:
        # Print confusion matrix terms
        print("=============================================");
        print("Confusion matrix for a binary classifier:");
        print("(number of) true positive (TP): ", TP)
        print("(number of) false positive (FP): ", FP)
        print("(number of) false negative (FN): ", FN)
        print("(number of) true negative (TN): ", TN)

        # Performance Evaluations
        print("Sensitivity: {0}%".format(round(Sensitivity, 4)*100))
        print("Specificity: {0}%".format(round(Specificity, 4)*100))
        print("Precision: {0}%".format(round(Precision, 4)*100))
        print("Accuracy: {0}%".format(round(Accuracy, 4)*100))
        print("MCC: {0}%".format(round(MCC, 4)*100))
        print("F1: {0}%".format(round(F1, 4)*100))
        print("=============================================\n");
    else:
        #print("\nI can't use thiss..!!!!");
        print("Result: TP={0}, FP={1}, FN={2}, TN={3}\n".format(TP,FP,FN,TN));
        
    
    # Store result in a file
    results = "(TP, FP, FN, TN={0}\t{1}\t{2}\t{3})\t\t fold{9}, classifier={8}, Sensitivity={4}, Specificity={5}, Accuracy={6}, MCC={7}".format(TP,FP,FN,TN,Sensitivity,Specificity,Accuracy,MCC,classifier_name, number_fold);
    fileName = saveFileName;
    readfile = open(fileName, "a")
    if(os.path.getsize(fileName) > 0):
       readfile.write("\n"+results)
    else:
       readfile.write(results)

    readfile.close()
    
    
# =============================================================================
# SET PATH, WORKING DIRECTORY AND READ DATA
# =============================================================================
# Set working directory
WORK_DIR = "D:/LAB PROJECT/JOURNALS/15 - Efflux v2/DATA/5 - BERT Features without SMOTE - CV/"
WORK_DIR = WORK_DIR+"TransportersBERT/"

# Output testing Cross-Validation
outputfile_matrix = WORK_DIR+"output cv bert/";
if not os.path.exists(outputfile_matrix):
    os.makedirs(outputfile_matrix)
    
# Set output path
outputfile_matrix = outputfile_matrix+"efflux-transport-transporters-bert.csv";
print("Ouput bert cv: "+outputfile_matrix);

# Read training data
read_train_class = pd.read_csv(WORK_DIR+'efflux_transport_train.csv')

# Print
read_train_class.head(5)

# Data and label/class
train_x_kfold, train_y_kfold = read_train_class.iloc[:,1:].values, read_train_class['class']

# Check and count pos an neg data
print('Original training data class #1 shape {}'.format(Counter(read_train_class['class'])))

# =============================================================================
# Gaussian Naive Bayes & 5-FOLD CROSS VALIDATION
# =============================================================================
print("=============================================");
print("Training Gaussian Naive Bayes classifier:");
kfold = StratifiedKFold(n_splits=5, shuffle=False)
fold = 0;
for train, test in kfold.split(train_x_kfold, train_y_kfold):
    fold = fold+1;
    print("Just counting number fold: {0}".format(fold));
    #print("Selected Training Number: {0}".format(train));
    #print("Selected Testing Number: {0}".format(test));
    
    # Select Train set from list of number
    train_x = train_x_kfold[np.array(train)]
    train_y = train_y_kfold[np.array(train)]
    
    # SMOTE for imbalance data
    k_neighbors=3 # int or object, optional (default=5)
    over_sample = SMOTE(k_neighbors=k_neighbors)
    train_x, train_y = over_sample.fit_sample(train_x, train_y)
    print('SMOTE: Resampled training dataset shape {}'.format(Counter(train_y)))
                
    # Select Test set from list of number
    test_x = train_x_kfold[np.array(test)]
    test_y = train_y_kfold[np.array(test)].tolist()
    
    # TRADITIONAL MACHINE LEARNING
    my_classifier = GaussianNB(priors=None)
    my_classifier.fit(train_x, train_y)
    pred_label = my_classifier.predict(test_x)
    # Print cofused Matrix
    ConfusionMatrixBinaryClassifier(fold, "Gaussian Naive Bayes", test_y, pred_label, outputfile_matrix);
    #print('Gaussian Naive BayesAccuracy: ', my_classifier.score(test_x, test_y))
    

print("=============================================");
print(" Training Random Forest classifier:");
kfold = StratifiedKFold(n_splits=5, shuffle=False)
fold = 0;
for train, test in kfold.split(train_x_kfold, train_y_kfold):
    fold = fold+1;
    print("Just counting number fold: {0}".format(fold));
    #print("Selected Training Number: {0}".format(train));
    #print("Selected Testing Number: {0}".format(test));
    
    # Select Train set from list of number
    train_x = train_x_kfold[np.array(train)]
    train_y = train_y_kfold[np.array(train)]
    
    # SMOTE for imbalance data
    k_neighbors=3 # int or object, optional (default=5)
    over_sample = SMOTE(k_neighbors=k_neighbors)
    train_x, train_y = over_sample.fit_sample(train_x, train_y)
    print('SMOTE: Resampled training dataset shape {}'.format(Counter(train_y)))
                
    # Select Test set from list of number
    test_x = train_x_kfold[np.array(test)]
    test_y = train_y_kfold[np.array(test)].tolist()
    
    # TRADITIONAL MACHINE LEARNING
    my_classifier = RandomForestClassifier(max_depth=10, n_estimators=30)
    my_classifier.fit(train_x, train_y)
    pred_label = my_classifier.predict(test_x)
    # Print cofused Matrix
    ConfusionMatrixBinaryClassifier(fold, "RandomForest", test_y, pred_label, outputfile_matrix);
    #print('RandomForest Accuracy: ', my_classifier.score(test_x, test_y))
    #print('RandomForest Feature Importances: ', my_classifier.feature_importances_)


# =============================================================================
#  Nearest Neighbors classifier
# =============================================================================
print("=============================================");
print(" Training Nearest Neighbors classifier:");
kfold = StratifiedKFold(n_splits=5, shuffle=False)
fold = 0;
for train, test in kfold.split(train_x_kfold, train_y_kfold):
    fold = fold+1;
    print("Just counting number fold: {0}".format(fold));
    #print("Selected Training Number: {0}".format(train));
    #print("Selected Testing Number: {0}".format(test));
    
    # Select Train set from list of number
    train_x = train_x_kfold[np.array(train)]
    train_y = train_y_kfold[np.array(train)]
    
    # SMOTE for imbalance data
    k_neighbors=3 # int or object, optional (default=5)
    over_sample = SMOTE(k_neighbors=k_neighbors)
    train_x, train_y = over_sample.fit_sample(train_x, train_y)
    print('SMOTE: Resampled training dataset shape {}'.format(Counter(train_y)))
                
    # Select Test set from list of number
    test_x = train_x_kfold[np.array(test)]
    test_y = train_y_kfold[np.array(test)].tolist()
    
    # TRADITIONAL MACHINE LEARNING
    n_neighbors = 100; # Optional (default = 5)
    weights = 'uniform' # str or callable, optional (default = 'uniform'), 'distance'
    algorithm = 'kd_tree' # {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
    leaf_size = 50; # Optional (default = 30)
    my_classifier = neighbors.KNeighborsClassifier()
    my_classifier.fit(train_x, train_y)
    pred_label = my_classifier.predict(test_x)
    # Print confusion matrix
    ConfusionMatrixBinaryClassifier(fold, "NearestNeighbors", test_y, pred_label, outputfile_matrix);
    #print('RandomForest Accuracy: ', my_classifier.score(test_x, test_y))
    #print('RandomForest Feature Importances: ', my_classifier.feature_importances_)


# =============================================================================
# Support Vector Machine - SVM - classifier: rbf
# =============================================================================
print("=============================================");
print("Training Support Vector Machine - SVM - classifier: rbf");
kfold = StratifiedKFold(n_splits=5, shuffle=False)
fold = 0;
for train, test in kfold.split(train_x_kfold, train_y_kfold):
    fold = fold+1;
    print("Just counting number fold: {0}".format(fold));
    #print("Selected Training Number: {0}".format(train));
    #print("Selected Testing Number: {0}".format(test));
    
    # Select Train set from list of number
    train_x = train_x_kfold[np.array(train)]
    train_y = train_y_kfold[np.array(train)]
    
    # SMOTE for imbalance data
    k_neighbors=3 # int or object, optional (default=5)
    over_sample = SMOTE(k_neighbors=k_neighbors)
    train_x, train_y = over_sample.fit_sample(train_x, train_y)
    print('SMOTE: Resampled training dataset shape {}'.format(Counter(train_y)))
                
    # Select Test set from list of number
    test_x = train_x_kfold[np.array(test)]
    test_y = train_y_kfold[np.array(test)].tolist()
    
    # TRADITIONAL MACHINE LEARNING
    kernel ='rbf'; # Optional (default='rbf') 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. If none is given, 'rbf' will be used.
    C = [1, 10, 100, 1000]  # SVM regularization parameter, optional (default=1.0)
    gamma = [0.00001, 0.0001, 0.001, 0.01] # float, optional (default='auto')
    for cost_ in C:
        for gamma_val in gamma:
            my_classifier = svm.SVC(kernel=kernel, gamma=gamma_val, C=cost_, probability=True)
            my_classifier.fit(train_x, train_y)
            pred_label = my_classifier.predict(test_x)
            # Print cofused Matrix
            ConfusionMatrixBinaryClassifier(fold, "svm_rbf c {0} g {1}".format(cost_, gamma_val), test_y, pred_label, outputfile_matrix);


      
