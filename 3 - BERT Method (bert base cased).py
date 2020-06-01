# -*- coding: utf-8 -*-

###############################################################################
# STEP 1 = LOAD PACKAGE
###############################################################################
import os
import pandas as pd
import numpy as np
import time        
from collections import Counter
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import StandardScaler

# EXTRACT DATA FILE FORMAT OF BERT
def read_extract_feature(list_file_paths):
  start_time = time.time()
  # temp store
  temp_store_id = [];
  temp_store_features = [];

  for file_path in list_file_paths:
    # Read data format 
    df_ori_data = pd.read_csv(file_path)

    # Default amino acids
    amino_acid_ =  ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'];

    # Sum/mean multiple row values of various columns grouped by 'residue'
    df_ori_sum = df_ori_data.groupby('residue', as_index=False).sum().copy()

    # Transpose first row as header
    df_ori_sum = df_ori_sum.set_index('residue').T

#    # Divided by sequence length
#    for amino in amino_acid_:
#       # Check if a column exists amino acid
#       if amino in df_ori_sum.columns:
#         df_ori_sum[amino] = df_ori_sum[amino].div((len(df_ori_data.index)), axis="index")
#       else:
#         df_ori_sum[amino] = 0;

    # Fill empty only
    for amino in amino_acid_:
      # Check if a column exists amino acid
      if amino not in df_ori_sum.columns:
        df_ori_sum[amino] = 0;

    df_ori_sum = df_ori_sum[amino_acid_].copy();

    # Transpose again
    df_ori_sum = df_ori_sum.transpose()

    # Pandas flatten a dataframe to a list (use .flatten() on the DataFrame)
    bert_feature_used = df_ori_sum.values.flatten();

    # Get protein id from path 
    prot_id = os.path.splitext(os.path.basename(""+file_path))[0];
    temp_store_id.append(prot_id);

    # Store
    temp_store_features.append(bert_feature_used);

  # Convert to dataframe
  df_results = pd.DataFrame(temp_store_features);
  df_results.insert(loc=0, column='accesion_id', value=temp_store_id) # insert a column at a specific column index=0

  # Timing
  print("[It takes {0} seconds to generate features]".format((time.time() - start_time)))

  return df_results;

# Standardization features
def scale_features(df_input):
    scaler = StandardScaler()
    test_lst = df_input.values.tolist()
    store_res = [];
    for lst in test_lst:
        np_arr = np.asarray(lst);
        X_transformed = scaler.fit_transform(np_arr.reshape(-1, 1))
        test_lst = X_transformed.reshape(1, -1).tolist()[0]
        store_res.append(test_lst);

    return pd.DataFrame(store_res);


###############################################################################
# STEP 2 = READ PATH AND SET BERT MODEL
###############################################################################
# Set bert model
pretrained_bert = "cased_L-12_H-768_A-12";


print("===========================================");
# Set working directory
WORK_DIR = "D:/LAB PROJECT/JOURNALS/15 - Efflux v2/DATA/4 - test and train CSV/";
OUTPUT_PATH_DIR = "D:/LAB PROJECT/JOURNALS/15 - Efflux v2/DATA/5 - BERT Features/";

# Training Data
read_train_class1 = pd.read_csv(WORK_DIR+"Data1 - efflux proteins [train].csv")
read_train_class1['data'] = 'efflux';
read_train_class2 = pd.read_csv(WORK_DIR+"Data2 - transport proteins [train].csv")
read_train_class3 = pd.read_csv(WORK_DIR+"Data3 - membrane proteins [train].csv")

# Testing Data
read_test_class1 = pd.read_csv(WORK_DIR+"Data1 - efflux proteins [test].csv")
read_test_class1['data'] = 'efflux';
read_test_class2 = pd.read_csv(WORK_DIR+"Data2 - transport proteins [test].csv")
read_test_class3 = pd.read_csv(WORK_DIR+"Data3 - membrane proteins [test].csv")


# Get accession_id list
lst_test_class1 = read_test_class1['accesion_id'].tolist()
lst_test_class2 = read_test_class2['accesion_id'].tolist()
lst_test_class3 = read_test_class3['accesion_id'].tolist()
lst_train_class1 = read_train_class1['accesion_id'].tolist()
lst_train_class2 = read_train_class2['accesion_id'].tolist()
lst_train_class3 = read_train_class3['accesion_id'].tolist()

# Set full path for bert model
bert_path = "D:/LAB PROJECT/JOURNALS/15 - Efflux v2/DATA/3 - All BERT Profiles/"+pretrained_bert+"/";

# List id with full path
nlst_test_class1 = [bert_path +x + '.bert_dstnc' for x in lst_test_class1]
nlst_test_class2 = [bert_path +x + '.bert_dstnc' for x in lst_test_class2]
nlst_test_class3 = [bert_path +x + '.bert_dstnc' for x in lst_test_class3]
nlst_train_class1 = [bert_path +x + '.bert_dstnc' for x in lst_train_class1]
nlst_train_class2 = [bert_path +x + '.bert_dstnc' for x in lst_train_class2]
nlst_train_class3 = [bert_path +x + '.bert_dstnc' for x in lst_train_class3]


###############################################################################
# Generate BERT features in dataframe
###############################################################################
df_test_class1 = read_extract_feature(nlst_test_class1);
df_test_class2 = read_extract_feature(nlst_test_class2);
df_test_class3 = read_extract_feature(nlst_test_class3);
df_train_class1 = read_extract_feature(nlst_train_class1);
df_train_class2 = read_extract_feature(nlst_train_class2);
df_train_class3 = read_extract_feature(nlst_train_class3);


# =============================================================================
# POSITIVE AND NEGATIVE DATASETS
# =============================================================================
# Set positive and negative classes
# Efflux vs transport
df_test_class1['class'] = 1  # 1 means positive instance, 0 means negative instance
df_train_class1['class'] = 1 
df_test_class2['class'] = 0
df_train_class2['class'] = 0 

#from sklearn.utils import resample
#df_efflux_bal_trns = resample(df_train_class1, n_samples=len(df_train_class2.index), random_state=0)

# Train
collect_train_class1 = pd.DataFrame();
collect_train_class1 = df_train_class1.append(df_train_class2)
# Test
collect_test_class1 = pd.DataFrame();
collect_test_class1 = df_test_class1.append(df_test_class2)

# Efllux vs membrane 
df_test_class1['class'] = 1  # 1 means positive instance, 0 means negative instance
df_train_class1['class'] = 1 
df_test_class3['class'] = 0
df_train_class3['class'] = 0 

#df_efflux_bal_memb = resample(df_train_class1, n_samples=len(df_train_class3.index), random_state=0)

# Train
collect_train_class2 = pd.DataFrame();
collect_train_class2 = df_train_class1.append(df_train_class3)
# Test
collect_test_class2 = pd.DataFrame();
collect_test_class2 = df_test_class1.append(df_test_class3)


###############################################################################
# CLASS 1 = EFFLUX VS TRANSPORTER
###############################################################################
# Read training Data
train_x = collect_train_class1.iloc[:,1:-1]
train_y = collect_train_class1.iloc[:,-1]
print('Original training dataset shape {}'.format(Counter(train_y)))

# Read Testing Data
test_x = collect_test_class1.iloc[:,1:-1]
test_y = collect_test_class1.iloc[:,-1]
print('Original testing dataset shape {}'.format(Counter(test_y)))

# Standardization features (feature scaling)
train_x = scale_features(train_x)
test_x = scale_features(test_x)


# SMOTE for imbalance data
k_neighbors=3 # int or object, optional (default=5)
over_sample = SMOTE(k_neighbors=k_neighbors)
train_x, train_y = over_sample.fit_sample(train_x, train_y)
print('SMOTE: Resampled training dataset shape {}'.format(Counter(train_y)))


# # Save Class 1 (EFFLUX VS TRANSPORTER)
OUTPUT_DIR_TARGET = "D:/LAB PROJECT/JOURNALS/15 - Efflux v2/DATA/5 - BERT Features/"+pretrained_bert+"/";
path_train = OUTPUT_DIR_TARGET+"efflux_transport_train.csv";
path_test = OUTPUT_DIR_TARGET+"efflux_transport_test.csv";

# To dataframe
df_data_train = pd.DataFrame(train_x)
df_data_test = pd.DataFrame(test_x)

# Insert class
df_data_train.insert(loc=0, column='class', value=train_y.values) # insert a column at a specific column index=0
df_data_test.insert(loc=0, column='class', value=test_y.values) # insert a column at a specific column index=0

# save
df_data_train.to_csv(path_train, encoding='utf-8', sep=',', index=False, header=True)
df_data_test.to_csv(path_test, encoding='utf-8', sep=',', index=False, header=True)


###############################################################################
# CLASS 2 = EFFLUX VS MEMBRANE
###############################################################################
# Read training Data
train_x = collect_train_class2.iloc[:,1:-1]
train_y = collect_train_class2.iloc[:,-1]
print('Original training dataset shape {}'.format(Counter(train_y)))

# Read Testing Data
test_x = collect_test_class2.iloc[:,1:-1]
test_y = collect_test_class2.iloc[:,-1]
print('Original testing dataset shape {}'.format(Counter(test_y)))

# Standardization features
train_x = scale_features(train_x)
test_x = scale_features(test_x)

# SMOTE for imbalance data
k_neighbors=3 # int or object, optional (default=5)
over_sample = SMOTE(k_neighbors=k_neighbors)
train_x, train_y = over_sample.fit_sample(train_x, train_y)
print('SMOTE: Resampled training dataset shape {}'.format(Counter(train_y)))


# # Save Class 2 (EFFLUX VS MEMBRANE)
OUTPUT_DIR_TARGET = "D:/LAB PROJECT/JOURNALS/15 - Efflux v2/DATA/5 - BERT Features/"+pretrained_bert+"/";
path_train = OUTPUT_DIR_TARGET+"efflux_membrane_train.csv";
path_test = OUTPUT_DIR_TARGET+"efflux_membrane_test.csv";

# To dataframe
df_data_train = pd.DataFrame(train_x)
df_data_test = pd.DataFrame(test_x)

# Insert class
df_data_train.insert(loc=0, column='class', value=train_y.values) # insert a column at a specific column index=0
df_data_test.insert(loc=0, column='class', value=test_y.values) # insert a column at a specific column index=0

# save
df_data_train.to_csv(path_train, encoding='utf-8', sep=',', index=False, header=True)
df_data_test.to_csv(path_test, encoding='utf-8', sep=',', index=False, header=True)



