# -*- coding: utf-8 -*-

import sys
import pandas as pd 
import time
import csv
import json
import numpy as np
import math
import os
import glob
import os.path
from os import path

# READ FASTA INPUT
def read_fasta_input(fastaSequenceInput):
    # Variables
    store_accesion_id = [];
    store_sequence_prot = [];
    store_seq_Length = [];
    
    data = fastaSequenceInput.replace('\n\n', '\n');
    getProtSeq = data.split(">")
    str_list = list(filter(None, getProtSeq)) # fastest
    
    for data_lst in str_list:
        try:
            each_prot = data_lst.split("\n")
            clear_prot = list(filter(None, each_prot)) # fastest
            # Get ID by first index and set to lowercase
            accesion_id = clear_prot[0];
            # Get sequence of protein by joining list from index
            get_sequence = "".join(clear_prot[1:len(clear_prot)]);
            get_sequence = get_sequence.replace('  ', ' ').replace(' ', '').replace('\t', '').replace('\n', '').replace('<br>', '');
            # Get sequence length
            get_seq_len = len(get_sequence);
            # Store
            store_accesion_id.append(accesion_id);
            store_sequence_prot.append(get_sequence);
            store_seq_Length.append(get_seq_len); 
        except:
            print("Found problem and skip proteins: {0}".format(data_lst));
    all_data = {'accesion_id' : store_accesion_id, 
                'sequence': store_sequence_prot,
                'length':store_seq_Length
               }
    return all_data; 

# N-GRAMS
def ngrams(input, n):
    # Cut string less than BERT max input 512
    if len(input) <= 510:
        input = input[0:len(input)];
    else:
        input = input[0:510];

    # Create a list and dataframe
    output = []

    # loop for each residues (+1 needs max the loop)
    for i in range(0, (len(input)+1)-n): # minus n means stop at final string
        # Cut for each n-gram
        g = input[i:i+n];

        # Score in list
        output.append(g);

    # Convert list to string
    joinstr = ' '.join(output);

    return joinstr;

# Cut string to list
def split_sequence_tolist(input_str, x):
    # Cut
    lst_res = [input_str[y-x:y] for y in range(x, len(input_str)+x, x)]
    
    return lst_res;

	
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

    # # Divided by sequence length
    # for amino in amino_acid_:
    #   # Check if a column exists amino acid
    #   if amino in df_ori_sum.columns:
    #     df_ori_sum[amino] = df_ori_sum[amino].div((len(df_ori_data.index)-2), axis="index")
    #   else:
    #     df_ori_sum[amino] = 0;

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
  

######################################################################
# INSTALLATION/CLONE BERT
# Change the working directory to the bert directory
######################################################################
import sys
os.system("git clone https://github.com/google-research/bert bert_repo");
if not 'bert_repo' in sys.path:
  sys.path += ['bert_repo']
  
import modeling
import optimization
import run_classifier
import run_classifier_with_tfhub
import tokenization
import tensorflow as tf
# import tfhub 
import tensorflow_hub as hub
import zipfile
import os


######################################################################
# **CALCULATE COSINE SIMILAR DISTANCE FOR EACH AMINO ACID**"""
######################################################################
# import pandas as pd 
import pandas as pd  
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np;
    
# List for an example
lst = [['a', 1, 2, 1, 2], 
       ['b', 2, 2, 2, 2], 
       ['s', 2, 2, 2, 2], 
       ['b', 2, 2, 2, 2], 
       ['c', 3, 2, 3, 2], 
       ['d', 4, 2, 4, 2], 
       ['f', 4, 2, 4, 2], 
       ['a', 5, 2, 5, 2], 
       ['c', 6, 2, 6, 2], 
       ['k', 7, 2, 7, 2], 
       ['b', 8, 2, 8, 2]] 

df_input_feature_embedding_vectors = pd.DataFrame(lst, columns =['residue', 'col1', 'col2', 'col3', 'col4']) 


###############################################################################
# FUNCTION TO CALCULATE COSINE DISTANCE FOR EACH AMINO ACID BASED ON INPUT STEPS
###############################################################################
"""
    This function is used to calculate cosine similarity distance for each amino acid
    based on the input embedding vectors in the dataframe.
    Parameters:
        @df_feature_vectors = input feature embedding vectors from BERT
        @step_down_slide_size = step_down_slide_size is the step size as the window 
            slides moved down across the dataframe called a stride (the defauld stride is 1).
            
    Output:
        The output will show all amino acids in the sequence and for each amino acid
        represents n best possible distance of amino acids.
"""
def calculate_cosine_distance_token_embeddings(df_feature_vectors, step_down_slide_size):
    # Get number rows
    row_numbers = len(df_feature_vectors.index);
    
    # Store all distances
    store_all_distances = [];
    
    if  row_numbers > step_down_slide_size:
      for index in range(0, row_numbers):
          start_index = 0;
          end_index = 0;
          if (index+step_down_slide_size) >= row_numbers:
              start_index = row_numbers - step_down_slide_size;
              end_index = row_numbers;
          else:
              start_index = index;
              end_index = ((step_down_slide_size+index)-1);
              
          # Select rows from index 0 with max length of step_down_slide_size
          data = df_feature_vectors.loc[start_index:end_index, :]
          #print(data);
          
          # Select target index row to list and skip first column is amino acid label
          aa_target = data.loc[index].values[1:];
          
          # Delete row with by index
          data = data.drop(index)
          
          #print("\nTarget index: "+str(index)+"");
          #print("Print list feature target:");
          #print(aa_target)
          
          ################################
          # Target amino acid token distance
          S1 = np.array(aa_target).reshape(1,-1) # Produces: [[ 0.58  0.76]]
          
          # Use .apply to apply cosine_similarity and save the new column
          data['distance'] = data.apply(lambda x: cosine_similarity(S1, np.array(x[1:]).reshape(1,-1))[0][0], axis=1)
            
          # Get all distances
          get_distances = data['distance'].values;
          
          # Store all distances
          store_all_distances.append(get_distances);
         
    # Return dataframe of cosine similar distance
    df_distance_result_ = pd.DataFrame(store_all_distances)     
    
    return df_distance_result_;     


## Call function for example
#df_distance_result = calculate_cosine_distance_token_embeddings(df_input_feature_embedding_vectors, 7);
##df_distance_result


################################################################################
# GET EMBEDDINGS FOR DATA INPUT DATA FILES
################################################################################
# Extract features for n-gram embeddings
def extract_fnl_feature_embeddings(path_bert_file, input_jsonl_file_path):
  start_time = time.time()
  
  # Temporary store variable
  temp_store_feature = [];
  embedding = []
  # Read JSONL files and append embedding vectors
  with open(input_jsonl_file_path) as f:
      for line in f:
        embedding.append(json.loads(line))
  # Print total rows data in test and train
  print("Number of protein embedings: "+str(len(embedding)))
    
  # Extract feature for each proteins here
  df_final_embding = pd.DataFrame();
  for row_index, get_prot_embedding in enumerate(embedding):
    # Temp variables
    store_token_amino_acid = [];
    store_token_embedding = [];
  
    # Get features
    features = embedding[row_index]["features"]
    # Extract amino acid tokens and vectors (token embedding)
    for index, feature in enumerate(features):
      token_amino_acid = feature["token"]

      # Order from original paper about layer (["layers"] ["index"] ["values"])
      # Index mens index of layer, ':' means select all layers
      token_embedding_layer0 = feature["layers"][0]["values"] # Last Layer
      token_embedding_layer1 = feature["layers"][1]["values"] # 2 Last Layer
      token_embedding_layer2 = feature["layers"][2]["values"] # 3 Last Layer
      token_embedding_layer3 = feature["layers"][3]["values"] # 4 Last Layer
      
      # Make list in list for all four layers
      token_embedding = [token_embedding_layer0, token_embedding_layer1, token_embedding_layer2, token_embedding_layer3];

      # Sum last 4 layers (sum of the last four layers)
      token_embedding = sum(map(np.array, token_embedding));
      #print(token_embedding);
      #print(token_amino_acid);
      
      # Store
      store_token_amino_acid.append(token_amino_acid);
      store_token_embedding.append(token_embedding);

      #print(f"{index}. token amino acid: {token_amino_acid}")
      #print(f" Protein embedding: {token_embedding[:]}")
      #print("\n")

    # Convert to dataframe (look like PSSM)
    data_bert = pd.DataFrame(store_token_embedding)
    #print(data_bert)

    # insert sequence labels as a new column at beginning of dataframe
    data_bert.insert(loc=0, column='residue', value=store_token_amino_acid) # Creat a new column represents all amino acids
	
    # Drop first and last row contains special token [CLS] and [SEP]
    data_bert = data_bert.drop(data_bert.index[0]) # First row
    data_bert = data_bert.drop(data_bert.index[len(data_bert)-1]) # Last row
    #print(data_bert)
	
    # All segment sequence store in the same dataframe
    df_final_embding = df_final_embding.append(data_bert);

  
  # Generate cosine distances
  df_final_embding = df_final_embding.reset_index(drop=True); # Need to reset index row
  df_feature_disntances = calculate_cosine_distance_token_embeddings(df_final_embding, 50);

  if len(df_feature_disntances.index) > 1:
    # Add residues
    df_feature_disntances.insert(loc=0, column='residue', value=df_final_embding['residue'].tolist())

    # Save to csv with protein id as a file name
    df_feature_disntances.to_csv(path_bert_file+'', index=False, header = True)
  
  else:
    print("This protein has short sequence: "+path_bert_file);

  
  # Timing
  print("[It takes {0} seconds to extract embedding features]".format((time.time() - start_time)))


################################################################################
# GENERATE JSONL OUTPUT EMBEDDING BERT
# Note: Auto detect for GPU when set use_tpu=False (training will fall on CPU or GPU)
# From the jsonl file you have last 4 layers outputs or -1,-2,-3,-4
# Get embeddings for input data classifiers from Google Colab terminal command
################################################################################
def generate_bert_output_embeddings_jsonl(df_data, bert_model_path, output_jsonl_file):
    start_time = time.time()
    get_path = bert_model_path;
    print("Bert Path: {0}".format(get_path));
	
    if bert_model_path.find('TransportersBERT') != -1: # result: -1
        check_point_path = bert_model_path+"/bert_model.ckpt-10000";
    elif bert_model_path.find('UniprotBERT') != -1: # result: -1
        check_point_path = bert_model_path+"/bert_model.ckpt-250000";
    else:
        check_point_path = bert_model_path+"/bert_model.ckpt";

    # Save temp dataframe and run bert embedding extractor (max_seq_length=384 or 512)
    df_data.to_csv('input.txt', index=False, header=False, quoting=csv.QUOTE_NONE)
    python_compiler = "/home/yzu1607b/anaconda3/envs/tensorflow/bin/python";
    os.system(python_compiler+" ./bert_repo/extract_features.py \
               --input_file=input.txt \
               --output_file="+output_jsonl_file+" \
               --vocab_file='"+bert_model_path+"/vocab.txt' \
               --bert_config_file='"+bert_model_path+"/bert_config.json' \
               --init_checkpoint='"+check_point_path+"' \
               --layers='-1,-2,-3,-4' \
               --max_seq_length=512 \
               --do_lower_case=False \
               --batch_size=8 \
               --use_tpu=False")

    #bert_output = pd.read_json("output.jsonl", lines=True)
    #bert_output.head()
    
    # Remove temp files
    os.system("rm input.txt")
    #os.system("rm output.jsonl")
    
    # Timing
    print("[It takes {0} seconds to generate JSONL file]".format((time.time() - start_time)))

	
#######################################################################
## READ INPUT DATA
#######################################################################
## Class 1 => ?
## Class 2 => ?
## Class 3 => ?
#
## Read input paramenters
#fasta_inputs = sys.argv[1]
#user_unique_id = sys.argv[2]
#user_email_address = sys.argv[3]
#user_pretrained_model = sys.argv[4]
#
## call functions
#get_array_fasta = read_fasta_input(fasta_inputs);
#
## Store in dataframe
#df_fasta_format = pd.DataFrame(get_array_fasta)     


# Read CSV
df_fasta_format = pd.read_csv("All protein sequence in the same format.csv")

print("Total fasta input: ", len(df_fasta_format.index))
print("Min len test: ", min(df_fasta_format['length'].tolist()))







# Create output directory
bert_output_path = "/home/yzu1607b/Semmy/Generate BERT/OUTPUT_BERT_PROFLES/";
if(path.exists(bert_output_path) == False):
	# Create dir to store new format
	os.mkdir(bert_output_path);
else:
	print("Directory is exist!!!");



###############################################################################
# LOOP DATA FOR EACH PROTEIN sequence & BERT PATH SETTINGS
# BECAUSE MAX LENGHT IS 510 (512), SO REPEATE THE PROCESS APPEND DATAFRAME LATER
###############################################################################
for index, row in df_fasta_format.iterrows():
	df_selected = df_fasta_format.iloc[index:index+1 , : ]; # for each row
	str_sequence = df_selected['sequence'].tolist()[0];
	
	# Split to max 510 amino acids (with 2 additional special tokens)
	lst_part_seq = split_sequence_tolist(str_sequence, 510);
	get_id = df_selected['accesion_id'].tolist()[0];
	
	# Check BERT profile exist or not
	bert_profile_path = bert_output_path+"bert/"+get_id+".bert_dstnc"; 
	
	#if not os.path.isfile(bert_profile_path):
	#	print("Not exist: "+get_id)
	
	if os.path.isfile(bert_profile_path):
		print("BERT feature exist: "+get_id)
	else:
		# Create dataframe for each proteins ID
		df_prot = pd.DataFrame({"sequence": lst_part_seq, 'accesion_id': get_id})
		
		# CREATE N-GRAMS DATA
		df_prot['1-grams'] = df_prot.apply(lambda x: ngrams(x['sequence'], 1), axis=1)
		df_prot['2-grams'] = df_prot.apply(lambda x: ngrams(x['sequence'], 2), axis=1)
		df_prot['3-grams'] = df_prot.apply(lambda x: ngrams(x['sequence'], 3), axis=1)
		print(df_prot);
    
		# BERT PATH SETTINGS
		#BERT_PRETRAINED_DIR = '/home/yzu1607b/BERT_MODELS/cased_L-12_H-768_A-12'
		#BERT_PRETRAINED_DIR = '/home/yzu1607b/BERT_MODELS/uncased_L-24_H-1024_A-16'
		#BERT_PRETRAINED_DIR = '/home/yzu1607b/BERT_MODELS/UniprotBERT'
		BERT_PRETRAINED_DIR = '/home/yzu1607b/BERT_MODELS/TransportersBERT'
		print('>>  BERT pretrained directory: '+BERT_PRETRAINED_DIR)
    
		# Generate output token embeding in Jsonl file from BERT model
		generate_bert_output_embeddings_jsonl(df_prot[['1-grams']], BERT_PRETRAINED_DIR, "test_class_input.jsonl");
		
		# Extract token embeddings and features from Jsonl file format of protein sequences
		extract_fnl_feature_embeddings(bert_profile_path, "test_class_input.jsonl"); 



















