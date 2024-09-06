import os
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
import glob

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    StoppingCriteriaList, 
    MaxLengthCriteria,
    StoppingCriteria,
)

from constants import SENTENCE_SEPARATOR, INSTRUCTION_SEPARATOR, RESPONSE_SEPARATOR
import torch

import math
import time
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
import re

import sys, json
from itertools import product



def data_prep(df, size,type):
    df = df[:size]
    df = df.fillna('NaN')
    df = df.astype(str)
    df = df.sort_values(by=['input','initial_response'])  
    texts = list(df["input"] + INSTRUCTION_SEPARATOR + df["initial_response"]+SENTENCE_SEPARATOR)
    return texts


    

def batch_responses_for_eval(model, tokenizer, texts_batch):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dot_id = tokenizer.encode(".")

    class0_texts = []
    class1_texts = []
    questions = []
    
    '''when generating, we will use the logits of right-most token to 
    predict the next token so the padding should be on the left'''
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error

    model_maxi_length = 512 
    
    inputs = tokenizer(texts_batch, padding=True, max_length=model_maxi_length, truncation=True, return_tensors="pt").to(device)

    maxi_new_tokens = 512 

    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        do_sample=False, 
        max_new_tokens = maxi_new_tokens, 
        eos_token_id = dot_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    
    init_sep = INSTRUCTION_SEPARATOR.replace(" ", '').replace("[", '').replace("]", '')
    aligned_sep = SENTENCE_SEPARATOR.replace(" ", '').replace("[", '').replace("]", '')
 
    for gen_text in generated_texts:
        generated_text = gen_text.replace("<|endoftext|>", '').strip()
        
        split_gen_text = re.split(' \['+init_sep+'\] | \['+aligned_sep+'\] ', generated_text)
        class0_text = None
        class1_text = None
        question = None
   
        if len(split_gen_text)>=3:
            question = split_gen_text[0]
            class0_text = split_gen_text[1]
            class1_text = split_gen_text[2]
        elif len(split_gen_text)==2:
            question = split_gen_text[0]
            class0_text = split_gen_text[1]
            class1_text = " "
        questions.append(question)
        class0_texts.append(class0_text)
        class1_texts.append(class1_text)
    return questions, class0_texts, class1_texts


def gather_responses(type,aligner_model_name,checkpoint_name,base_model,aligned_dfs_path):
    final_aligned_df_path = aligned_dfs_path+"final/"
    os.makedirs(final_aligned_df_path, exist_ok=True)
  
    df_list = []   
    for file_data in glob.glob(aligned_dfs_path + '*.csv'):
        batch_data_df = pd.read_csv(file_data)
        df_list.append(batch_data_df)  
    data_df = pd.concat(df_list)
    data_df = data_df.astype(str)
    
    # save final df to final_aligned_df_path and do for all aligner types
    data_df.to_csv(final_aligned_df_path+type+"_"+base_model+"_aligned_data_using_"+aligner_model_name+"_aligner.csv", index=False)
    return "Done combining all responses."



def get_eval_results(model, tokenizer, texts,batch_size,aligned_dfs_path,checkpoint_name,type,aligner_model_name,base_model):
    
    saved_files = glob.glob(aligned_dfs_path + '*.csv')
    start_index = None
    if len(saved_files)!=0:
        saved_files_ids = [int(saved_file.split(".")[1].split("_")[-1]) for saved_file in saved_files]
        saved_files_ids.sort()
        start_index = saved_files_ids[-1]+batch_size
    else:
        start_index = 0
    
    end_index = len(texts)
    
    for index in range(start_index, end_index, batch_size):
        generated_texts0 = []
        generated_texts1 = []
        questions_list = []

        texts_batch = texts[index : index + batch_size]

        questions_batch, class0_batch, class1_batch = batch_responses_for_eval(model, tokenizer, texts_batch)
        questions_list.extend(questions_batch)
        generated_texts0.extend(class0_batch)
        generated_texts1.extend(class1_batch)
        
        data_df = pd.DataFrame(questions_list, columns=['input'])
        data_df["initial_response"] = generated_texts0
        data_df["aligned_response"] = generated_texts1
        data_df.to_csv(aligned_dfs_path+"df_"+str(checkpoint_name)+"_index_"+str(index)+".csv", index=False)
        
    if any(dir.startswith('final') for dir in os.listdir(aligned_dfs_path)):
        print("Completed!")
    else:
        gather_responses(type,aligner_model_name,checkpoint_name,base_model,aligned_dfs_path)
    return "DONE."
             


def generate_responses_using_aligners(type=None, batch_size=None, eval_set_size=None, 
                                      aligner_model_name=None,base_model=None,checkpoint_id=None,dataset=None):
    unaligned_df = pd.read_csv("./data_unaligned/"+dataset+"/"+base_model+"/final/"+dataset+"_initial_response_"+base_model+"_noprompt.csv")

    aligned_dfs_path = "./data_aligned_individual/"+dataset+"/"+type+"/"+aligner_model_name+"/checkpoint-"+str(checkpoint_id)+"/"+base_model+"/dfs/"
     
    path = "../aligner-training/saved_checkpoints/"+type+"/"+aligner_model_name+"/"  

    os.makedirs(aligned_dfs_path, exist_ok=True)
    
    texts = data_prep(unaligned_df, eval_set_size,type)
    
    path_checkpoint = path+"checkpoint-"
    
    model_paths = [checkpoint_id]

    model_paths.sort()
    sorted_modelpaths = [path_checkpoint+str(sp) for sp in model_paths]

    for modelpath in sorted_modelpaths:
        tokenizer = AutoTokenizer.from_pretrained(modelpath, padding_side="left",device_map='auto')
        model = AutoModelForCausalLM.from_pretrained(modelpath,device_map='auto')
        checkpoint_name = modelpath.split("/")[-1]

        '''Generate responses using aligners'''
        get_eval_results(model, tokenizer, texts,batch_size,aligned_dfs_path,checkpoint_name,type,aligner_model_name,base_model)
        print(modelpath)
    
    

if __name__ == "__main__":
    batch_size_ = [64]
    eval_set_size_ = [15000] 
    checkpoint_ids = [2500]
    datasets = ["synthetic_mixed", "beaverTails"]
    types = ["ethical", "factuality", "helpful"]
    aligner_model_names = ["gpt2", "pythia", "redpajama", "phi2"]
    base_models = ["falcon-40b", "llama-2-13b", "llama-2-70b"]

    grid = list(product(base_models, aligner_model_names, types, datasets, checkpoint_ids, eval_set_size_, batch_size_))
    
    i = int(float(sys.argv[1]))
    print("print i")
    print(sys.argv[1])
    base_model, aligner_model_name, type, dataset, checkpoint_id, eval_set_size, batch_size = grid[i]    
    print("Print Model Name!!!")
    print(aligner_model_name)

    generate_responses_using_aligners(type=type, batch_size=batch_size, eval_set_size=eval_set_size, aligner_model_name=aligner_model_name,
                                      base_model=base_model,checkpoint_id=checkpoint_id, dataset=dataset)



