# -*- coding: utf-8 -*-

# Packages
import pandas as pd;

# Imput file
dir_path = "D:/LAB PROJECT/JOURNALS/15 - DeepEfflux v2/DATA/2 - Remove similarity/";
file_path = dir_path+"Fasta file for representative sequences at 30% identity.fasta";

# Variables
store_accesion_id = [];
store_decription = [];
store_sequence_prot = [];
store_seq_Length = [];
store_data_number = []

#==============================================================================
# Read file data
#==============================================================================
with open(file_path, 'r') as file:
    data = file.read()
    data = data.replace('\n\n', '\n');
    # Split ">"    
    getProtSeq = data.split(">_")
    len(getProtSeq);
    
    # Remove empty strings from a list of strings
    str_list = list(filter(None, getProtSeq)) # fastest
    
    for data_lst in str_list:
        #print(data_lst);
        # Split for each part in each protein
        try:
            each_prot = data_lst.split("\n")
            clear_prot = list(filter(None, each_prot)) # fastest
            
            # Get ID by first index and set to lowercase
            get_ori_id_prot = clear_prot[0];
            
            # get accesion_id format split
            # Example: gnl|TC-DB|222437377|1.M.1.3.10 hypothetical protein SADFL11_1342 [Labrenzia alexandrii DFL-11]
            splt_get_ori_id = get_ori_id_prot.split("|");
            
            # Get accesion id by getting a string berfore a specific whitespace (ex, 1.M.1.3.10)
            accesion_id = splt_get_ori_id[1];
            get_data_number = splt_get_ori_id[0].replace('_sp', '') #"format like this 2_" index 1 to get value 2
            get_dec = splt_get_ori_id[2];
            
            # Get sequence of protein by joining list from index
            get_sequence = "".join(clear_prot[1:len(clear_prot)]);
            get_sequence = get_sequence.replace(' ', '').replace('\t', '');
            
            # Get sequence length
            get_seq_len = len(get_sequence);
            
            if get_seq_len > 20:
                # Store
                store_accesion_id.append(accesion_id);
                store_sequence_prot.append(get_sequence);
                store_seq_Length.append(get_seq_len); 
                store_decription.append(get_dec);
                store_data_number.append(get_data_number);
        except:
            print("Skipp proteins: {0}".format(data_lst));

print("-- FINISHED READ FILE--");                
#==============================================================================
# SAVE IN FILE
#==============================================================================
# Create an dataframe
all_data = {'data': store_data_number,
            'accesion_id' : store_accesion_id, 
            'description': store_decription,
            'length':store_seq_Length,
            'sequence': store_sequence_prot
            }

# Create
df = pd.DataFrame(all_data)

# Sort pandas dataframe from one column
df = df.sort_values('data')

# Reset index in a pandas data frame
df = df.reset_index(drop=True)

# Select only length >= 50
df = df.loc[df['length'] >= 50]
#print(df.loc[df['length'] >= 50])

# For Seven human coronaviruses
# Data1: Human coronavirus 229E, taxid:11137 (HCoV-229E) (1007 proteins)
data_num_1 = df.loc[(df['data'] == 'mfs') | (df['data'] == 'smr') | (df['data'] == 'mate') | (df['data'] == 'rnd') | (df['data'] == 'abc')]
data_num_1.to_csv (dir_path+'Data1 - efflux proteins [converted 30%].csv', index = None, header=True) # Don't forget to add '.csv' at the end of the path

# Data2: Human coronavirus HKU1 (HCoV-HKU1), taxid:290028 (751 proteins)
data_num_2 = df.loc[df['data'] == "transport"]
data_num_2 = data_num_2.loc[data_num_2['length'] <= 1000]
data_num_2 = data_num_2[~data_num_2.sequence.str.contains('X',case=False)]
data_num_2.to_csv (dir_path+'Data2 - transport proteins [converted 30%].csv', index = None, header=True) # Don't forget to add '.csv' at the end of the path

# Data3: Human coronavirus NL63 (HCoV-NL63), taxid:277944 (1318 proteins)
data_num_3 = df.loc[df['data'] == "membrane"]
data_num_3 = data_num_3.loc[data_num_3['length'] <= 1000]
data_num_3 = data_num_3[~data_num_3.sequence.str.contains('X',case=False)]
data_num_3.to_csv (dir_path+'Data3 - membrane proteins [converted 30%].csv', index = None, header=True) # Don't forget to add '.csv' at the end of the path

print("-- DONE EXPORT FILE CSV--");   
