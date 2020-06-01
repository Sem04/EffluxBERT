#!/usr/bin/env python

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn import metrics
	
"""
	A receiver operating characteristic curve, or ROC curve, is a graphical plot that illustrates the diagnostic ability of a 
	binary classifier system as its discrimination threshold is varied. The ROC curve is created by plotting the true positive rate (TPR) 
	against the false positive rate (FPR) at various threshold settings. (Wikipedia)
"""
def plotROC(test_y, score, methodname, file_name_roc):
	fpr,tpr,threshold = roc_curve(test_y, score)
	roc_auc_ind = metrics.auc(fpr, tpr, reorder=True);# reorder=True
	plt.figure()
	lw = 2
	plt.figure(figsize=(10,10))
	plt.plot(fpr, tpr, color='red',lw=lw, label='{0} (AUC={1})'.format(methodname, round(roc_auc_ind, 3)))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.ylabel('False Positive Rate (Sensitivity)')
	plt.xlabel('True Positive Rate (1-Specificity)')
	plt.title('ROC Curve Analysis')
	plt.legend(loc="lower right")
	plt.savefig(file_name_roc)
	
	
"""
	ROC curve for models of all classfiers
"""
def plotROC_all_classifiers(set_title, single_classes, pred_prob_svm, pred_prob_rfc, pred_prob_gnb, pred_prob_knn, file_name_roc):
	# FPR, TPR
	fpr_svm, tpr_svm, threshold_svm = roc_curve(single_classes, pred_prob_svm)
	fpr_rb, tpr_rb, threshold_rb = roc_curve(single_classes, pred_prob_rfc)
	fpr_gnb, tpr_gnb, threshold_gnb = roc_curve(single_classes, pred_prob_gnb)
	fpr_knn, tpr_knn, threshold_knn = roc_curve(single_classes, pred_prob_knn)
	
	# Get Area under curve (AUC)
	roc_auc_ind_svm = metrics.auc(fpr_svm, tpr_svm);# reorder=True
	roc_auc_ind_rb = metrics.auc(fpr_rb, tpr_rb);# reorder=True
	roc_auc_ind_gnb = metrics.auc(fpr_gnb, tpr_gnb);# reorder=True
	roc_auc_ind_knn = metrics.auc(fpr_knn, tpr_knn);# reorder=True
	
	plt.figure()
	lw = 2
	plt.figure(figsize=(10,10))
	plt.plot(fpr_gnb, tpr_gnb, color='yellow',lw=lw, label='GNB (AUC={0})'.format(round(roc_auc_ind_gnb, 3)))
	plt.plot(fpr_rb, tpr_rb, color='blue',lw=lw, label='RFC (AUC={0})'.format(round(roc_auc_ind_rb, 3)))
	plt.plot(fpr_knn, tpr_knn, color='green',lw=lw, label='KNN (AUC={0})'.format(round(roc_auc_ind_knn, 3)))
	plt.plot(fpr_svm, tpr_svm, color='red',lw=lw, label='SVM-RBF (AUC={0})'.format(round(roc_auc_ind_svm, 3)))
	
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.ylabel('False Positive Rate (Sensitivity)')
	plt.xlabel('True Positive Rate (1-Specificity)')
	plt.title(''+set_title)
	plt.legend(loc="lower right")
	plt.savefig(file_name_roc)	
	
	

	
	