# -*- coding: utf-8 -*-

import pandas as pd;

from sklearn.utils import shuffle

# READ ALL CSV AFTER USING CD-HIT

# Seven human coronaviruses
dir_path = "D:/LAB PROJECT/JOURNALS/15 - Efflux v2/DATA/2 - Remove similarity/";
df_data3_1 = pd.read_csv(dir_path+"Data1 - efflux proteins [converted 30%].csv")
df_data3_2 = pd.read_csv(dir_path+"Data2 - transport proteins [converted 30%].csv")
df_data3_3 = pd.read_csv(dir_path+"Data3 - membrane proteins [converted 30%].csv")

# save all fasta csv
dir_path = "D:/LAB PROJECT/JOURNALS/15 - Efflux v2/DATA/3 - test and train CSV/";

# For Data1 - efflux proteins
data1 = df_data3_1.loc[df_data3_1['length'] > 50] # Select only length >= 50
data1 = shuffle(data1).reset_index(drop=True) # Shuffle DataFrame rows

data1_mfs = df_data3_1.loc[df_data3_1['data'] == 'mfs']
data1_smr = df_data3_1.loc[df_data3_1['data'] == 'smr']
data1_mate = df_data3_1.loc[df_data3_1['data'] == 'mate']
data1_rnd = df_data3_1.loc[df_data3_1['data'] == 'rnd']
data1_abc = df_data3_1.loc[df_data3_1['data'] == 'abc']

test_mfs = data1_mfs.iloc[0:63, :]
test_smr = data1_smr.iloc[0:1, :]
test_mate = data1_mate.iloc[0:4, :]
test_rnd = data1_rnd.iloc[0:3, :]
test_abc = data1_abc.iloc[0:42, :]

train_mfs = data1_mfs.iloc[63:, :]
train_smr = data1_smr.iloc[1:, :]
train_mate = data1_mate.iloc[4:, :]
train_rnd = data1_rnd.iloc[3:, :]
train_abc = data1_abc.iloc[42:, :]

data1_test = test_mfs.append(test_smr).append(test_mate).append(test_rnd).append(test_abc);
data1_train = train_mfs.append(train_smr).append(train_mate).append(train_rnd).append(train_abc);

data1_test.to_csv (dir_path+'Data1 - efflux proteins [test].csv', index = None, header=True) # Don't forget to add '.csv' at the end of the path
data1_train.to_csv (dir_path+'Data1 - efflux proteins [train].csv', index = None, header=True) # Don't forget to add '.csv' at the end of the path


# For Data2 - transport proteins
data2 = df_data3_2.loc[df_data3_2['length'] > 50] # Select only length >= 50
data2 = shuffle(data2).reset_index(drop=True) # Shuffle DataFrame rows
data2_test = data2.iloc[0:454, :]
data2_train = data2.iloc[454:, :]
data2_test.to_csv (dir_path+'Data2 - transport proteins [test].csv', index = None, header=True) # Don't forget to add '.csv' at the end of the path
data2_train.to_csv (dir_path+'Data2 - transport proteins [train].csv', index = None, header=True) # Don't forget to add '.csv' at the end of the path


# For Data2 - membrane proteins
data3 = df_data3_3.loc[df_data3_3['length'] > 50] # Select only length >= 50
data3 = shuffle(data3).reset_index(drop=True) # Shuffle DataFrame rows
data3_test = data3.iloc[0:1084, :]
data3_train = data3.iloc[1084:, :]
data3_test.to_csv (dir_path+'Data3 - membrane proteins [test].csv', index = None, header=True) # Don't forget to add '.csv' at the end of the path
data3_train.to_csv (dir_path+'Data3 - membrane proteins [train].csv', index = None, header=True) # Don't forget to add '.csv' at the end of the path





