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



def prompt_formatter(input, init_response):
    prompt = 'BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair to make it more helpful and harmless: {question} | {answer} ASSISTANT:'
    formated_input = prompt.format(
        question=input,
        answer=init_response
        )
    return formated_input
    

def generate_test_responses(model, tokenizer, test_examples_path, qualitative_responses_path):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    df = pd.read_csv(test_examples_path)
    texts = [prompt_formatter(list(df["input"])[i], list(df["initial_response"])[i]) for i in range(len(df))]  

    generated_texts = []

    for text in texts:
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        output_ids = model.generate(input_ids, max_new_tokens=2048)[0]
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        generated_texts.append(generated_text)
    print(generated_texts)
    
    generated_df = pd.DataFrame(generated_texts, columns=["generated_responses"])
    generated_df.to_csv(qualitative_responses_path+"generated_test_output.csv", index=False)
    return "Qualitative Eval DONE."
 
    

def batch_responses_for_eval(model, tokenizer, texts_batch, inputs, class0_responses):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    class0_texts = []
    class1_texts = []
    questions = []
    
    for i in range(len(texts_batch)): 
        input_ids = tokenizer.encode(texts_batch[i], return_tensors='pt').to(device)
        output_ids = model.generate(input_ids, max_new_tokens=2048)[0]
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        question = inputs[i]
        class0_text = class0_responses[i]
        class1_text = generated_text.split("ASSISTANT:")[-1].strip()
        questions.append(question)
        class0_texts.append(class0_text)
        class1_texts.append(class1_text)
    return questions, class0_texts, class1_texts


def gather_responses(type,aligner_model_name,base_model,aligned_dfs_path):
    final_aligned_df_path = "./data_aligned/"+type+"/"+aligner_model_name+"/"+base_model+"/dfs/final/"
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



def generate_responses(model, tokenizer, unaligned_df,batch_size,aligned_dfs_path,type,aligner_model_name,base_model):
    saved_files = glob.glob(aligned_dfs_path + '*.csv')
    start_index = None
    if len(saved_files)!=0:
        saved_files_ids = [int(saved_file.split(".")[1].split("_")[-1]) for saved_file in saved_files]
        saved_files_ids.sort()
        start_index = saved_files_ids[-1]+batch_size
    else:
        start_index = 0
    
    end_index = len(unaligned_df)
    
    for index in range(start_index, end_index, batch_size):
        
        print("Print batch size!")   
        print(batch_size)
        df_batch = unaligned_df[index : index + batch_size]
        texts_batch = [prompt_formatter(list(df_batch["input"])[i], list(df_batch["initial_response"])[i]) for i in range(len(df_batch))]  
        inputs = list(df_batch["input"])
        class0_responses = list(df_batch["initial_response"])
        questions_batch, class0_batch, class1_batch = batch_responses_for_eval(model, tokenizer, texts_batch, inputs, class0_responses)
        
        print("At index ", index)
        data_df = pd.DataFrame(questions_batch, columns=['input'])
        data_df["initial_response"] = class0_batch
        data_df["aligned_response"] = class1_batch
        data_df.to_csv(aligned_dfs_path+"df_index_"+str(index)+".csv", index=False)
        
    gather_responses(type,aligner_model_name,base_model,aligned_dfs_path)
    return "DONE."
             


def generate_responses_using_PKU_aligner(type=None, batch_size=None, eval_set_size=None, aligner_model_name=None,base_model=None):
    test_examples_path = "./qualitative_data/qualitative_eval_data.csv" 

    unaligned_df = pd.read_csv("./data_unaligned/"+type+"/"+base_model+"/final/"+type+"_initial_response_"+base_model+"_noprompt.csv")

    eval_set_size = min(eval_set_size, len(unaligned_df))
    unaligned_df = unaligned_df[:eval_set_size]
    
    qualitative_responses_path = "./data_aligned/"+type+"/"+aligner_model_name+"/qualitative_examples/"
    aligned_dfs_path = "./data_aligned/"+type+"/"+aligner_model_name+"/"+base_model+"/dfs/"
    
    os.makedirs(qualitative_responses_path, exist_ok=True)
    os.makedirs(aligned_dfs_path, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained('aligner/aligner-7b-v1.0',device_map='auto', cache_dir='./cache/')
    tokenizer =AutoTokenizer.from_pretrained('aligner/aligner-7b-v1.0',use_fast=False, cache_dir='./cache/')

    ''' Generate responses for qualitative examples'''
    generate_test_responses(model, tokenizer, test_examples_path, qualitative_responses_path)
    '''Generate test data responses using aligner'''
    generate_responses(model, tokenizer, unaligned_df, batch_size,aligned_dfs_path,type,aligner_model_name,base_model)


if __name__ == "__main__":
    batch_size_ = [128]
    eval_set_size_ = [15000] 
    types = ["synthetic_mixed","beaverTails"]
    aligner_model_names = ["pku_aligner"] 
    base_models = ["falcon-40b", "llama-2-13b", "llama-2-70b"]

    grid = list(product(base_models, aligner_model_names, types, eval_set_size_, batch_size_))
    
    i = int(float(sys.argv[1]))
    print("print i")
    print(sys.argv[1])
    base_model, aligner_model_name, type, eval_set_size, batch_size = grid[i]    
    print("Print Model Name!!!")
    print(aligner_model_name)

    generate_responses_using_PKU_aligner(type=type, batch_size=batch_size, eval_set_size=eval_set_size,
                                         aligner_model_name=aligner_model_name,base_model=base_model)



