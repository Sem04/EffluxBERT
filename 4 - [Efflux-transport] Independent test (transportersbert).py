# -*- coding: utf-8 -*-

# =============================================================================
# Load Packages
# =============================================================================
import pandas as pd 
import numpy as np
import math
from sklearn import svm
from sklearn import neighbors
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import os

##########################################################
# Confusion matrix
##########################################################
def ConfusionMatrixBinaryClassifier(classifier_name, test_label_original, test_label_predicted, saveFileName):
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
    results = "(TP, FP, FN, TN={0}\t{1}\t{2}\t{3}), classifier={8}, Sensitivity={4}, Specificity={5}, Accuracy={6}, MCC={7}".format(TP,FP,FN,TN,Sensitivity,Specificity,Accuracy,MCC,classifier_name);
    fileName = saveFileName;
    readfile = open(fileName, "a")
    if(os.path.getsize(fileName) > 0):
       readfile.write("\n"+results)
    else:
       readfile.write(results)

    readfile.close()

# =============================================================================
# SET PATH AND WORKING DIRECTORY
# =============================================================================
# Set working directory
WORK_DIR = "D:/LAB PROJECT/JOURNALS/15 - Efflux v2/DATA/5 - BERT Features/"
WORK_DIR = WORK_DIR+"TransportersBERT/"

# =============================================================================
# READ DATA
# =============================================================================
# Read testing data
read_test_class = pd.read_csv(WORK_DIR+'efflux_transport_test.csv')

# Read training data
read_train_class = pd.read_csv(WORK_DIR+'efflux_transport_train.csv')

# Output testing
outputfile_matrix = WORK_DIR+"output independent/efflux.transport.transportersbert.output.csv";
print(outputfile_matrix);

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


# Check and count pos an neg data
print('Original training data class #1 shape {}'.format(Counter(read_train_class['class'])))
print('Original testing data class #1 shape {}'.format(Counter(read_test_class['class'])))

# =============================================================================
# TRADITIONAL MACHINE LEARNING ALGORITHMS
# =============================================================================
print("Training Gaussian Naive Bayes classifier:");
my_classifier = GaussianNB(priors=None)
my_classifier.fit(train_x, train_y)
pred_label = my_classifier.predict(test_x)
# Print cofused Matrix
ConfusionMatrixBinaryClassifier("GaussianNaiveBayes", test_y, pred_label, outputfile_matrix);

print(" Training Random Forest classifier:");
my_classifier = RandomForestClassifier(max_depth=50, n_jobs=50, random_state=0, n_estimators=100)
my_classifier.fit(train_x, train_y)
pred_label = my_classifier.predict(test_x)
# Print cofused Matrix
ConfusionMatrixBinaryClassifier("RandomForest", test_y, pred_label, outputfile_matrix);

print(" Training Nearest Neighbors classifier:");
n_neighbors = 100; # Optional (default = 5)
weights = 'uniform' # str or callable, optional (default = 'uniform'), 'distance'
algorithm = 'kd_tree' # {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
leaf_size = 50; # Optional (default = 30)
my_classifier = neighbors.KNeighborsClassifier()
my_classifier.fit(train_x, train_y)
pred_label = my_classifier.predict(test_x)
# Print cofused Matrix
ConfusionMatrixBinaryClassifier("NearestNeighbors", test_y, pred_label, outputfile_matrix);

print("Training Support Vector Machine - SVM - classifier: rbf");
kernel ='rbf'; # Optional (default='rbf') 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. If none is given, 'rbf' will be used.
C = [1, 10, 100, 1000]  # SVM regularization parameter, optional (default=1.0)
gamma = [0.00001, 0.0001, 0.001, 0.01] # float, optional (default='auto')
for cost_ in C:
    for gamma_val in gamma:
        my_classifier = svm.SVC(kernel=kernel, gamma=gamma_val, C=cost_, probability=True)
        my_classifier.fit(train_x, train_y)
        pred_label = my_classifier.predict(test_x)
        # Print cofused Matrix
        ConfusionMatrixBinaryClassifier("svm_rbf c {0} g {1}".format(cost_, gamma_val), test_y, pred_label, outputfile_matrix);


# Save the model to local
import pickle
path_dir = "D:/LAB PROJECT/JOURNALS/15 - Efflux v2/DATA/6 - Classifier Models/";
filenamemodel = path_dir+'efflux-transport-transportersbert.sav'
pickle.dump(my_classifier, open(filenamemodel, 'wb'))


    