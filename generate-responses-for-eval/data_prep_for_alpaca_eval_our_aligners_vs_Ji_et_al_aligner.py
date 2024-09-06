import os
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
import glob

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
from constants import SENTENCE_SEPARATOR, INSTRUCTION_SEPARATOR, RESPONSE_SEPARATOR,SENTENCE_END


def data_prep(df, size):
    df = df.fillna('NaN')
    df = df.astype(str)
    
    if len(df.columns)>2:
        end = SENTENCE_END.replace(" ", '').replace("[", '').replace("]", '')
        aligned_responses_ = list(df["aligned_response"])
        class1_responses = []
        for aligned in aligned_responses_:
            if SENTENCE_END in aligned:  
                aligned_response = re.split(' \['+end+'\]', aligned)[0]
                class1_responses.append(aligned_response)
            else:
                class1_responses.append(aligned)
                
        df["aligned_response"] = class1_responses
    return df 


def prep_data_for_alpaca_eval(eval_set_size=None, aligner_model_name=None,
                             base_model=None,checkpoint_id=None, evaluator=None,batch_size=None,type=None, dataset=None, category=None):
    all_combined_dfs = []
    combined_results_df = None
    aligned_by_aligners_df = None
    if category == "squad":
        aligned_by_aligners_df = pd.read_csv("./data_aligned/"+dataset+"/"+aligner_model_name+
                              "/checkpoint-"+str(checkpoint_id)+"/"+base_model+"/dfs/final/"+dataset+"_"+base_model+
                              "_aligned_data_using_"+aligner_model_name+"_aligner.csv")

    else:
        aligned_by_aligners_df = pd.read_csv("../eval-data-generation/data_aligned_individual/"+dataset+"/"+type+"/"+aligner_model_name+
                      "/checkpoint-"+str(checkpoint_id)+"/"+base_model+"/dfs/final/"+type+"_"+base_model+
                      "_aligned_data_using_"+aligner_model_name+"_aligner.csv")
        
        aligned_by_aligners_df = data_prep(aligned_by_aligners_df, eval_set_size)

        aligned_by_aligners_df = aligned_by_aligners_df[["input", "aligned_response"]]
        
    aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("</s>","")
    aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("<s>","")
    aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].apply(lambda x: str(x).strip())

    aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].str.replace("</s>","")
    aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].str.replace("<s>","")
    aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].apply(lambda x: str(x).strip())
    
    aligned_by_aligners_df.columns = ["input", "aligned_response_by_our_aligner"]

    aligned_by_pku_aligner_df = pd.read_csv("../eval-data-generation/data_aligned/"+dataset+"/pku_aligner/"+base_model+
                                             "/dfs/final/"+dataset+"_"+base_model+"_aligned_data_using_pku_aligner_aligner.csv")
    
    aligned_by_pku_aligner_df = aligned_by_pku_aligner_df[["input", "aligned_response"]]
    aligned_by_pku_aligner_df.columns = ["input", "aligned_response_by_PKU_aligner"]
   
    cols = ['input']
    combined_df = aligned_by_pku_aligner_df.merge(aligned_by_aligners_df, 'outer',on=cols, indicator=True)
    combined_df = combined_df[combined_df['_merge'] == 'both']
    combined_df.drop_duplicates(subset=['input'], keep='first', inplace=True, ignore_index=True)
    
    final_combined_df = combined_df[["input","aligned_response_by_PKU_aligner","aligned_response_by_our_aligner"]]
    
    final_combined_df = final_combined_df[~(final_combined_df["aligned_response_by_our_aligner"].str.contains("Nothing"))]
    
    final_combined_df.columns = ["input", "aligned_response_by_PKU_aligner", "aligned_response_by_"+aligner_model_name]
    final_combined_df = final_combined_df.astype(str)
    combined_results_df = final_combined_df[:eval_set_size]
    
    combined_results_df = combined_results_df.fillna('NaN')
    combined_results_df = combined_results_df.astype(str)
    combined_results_df = combined_results_df.dropna()
    
    path = None
    if category == "squad":
        path = "./data_for_alpaca_eval/our_aligners_vs_Ji_et_al_aligner/"+category+"/"+dataset+"/"+aligner_model_name+"/"+base_model+"/"

    else:
        path = "./data_for_alpaca_eval/our_aligners_vs_Ji_et_al_aligner/"+category+"/"+dataset+"/"+type+"/"+aligner_model_name+"/"+base_model+"/"
    os.makedirs(path, exist_ok=True)
    combined_results_df.to_csv(path+"df.csv", index=False)

    '''Folders where results will be stored'''
    results_folder = path + "/our_aligner_vs_Ji_et_al_aligner/"
    os.makedirs(results_folder, exist_ok=True)

    combined_results_df = pd.read_csv(path+'df.csv').dropna().reset_index(drop=True)
    
    combined_results_df = combined_results_df.sample(n=800)
   
    pku_responses_df = combined_results_df[["input", "aligned_response_by_PKU_aligner"]]
    pku_responses_df.columns = ["instruction", "output"]
    pku_responses_df["generator"] = "PKU_aligner"
    pku_responses_df["dataset"] = dataset
    pku_responses_df["datasplit"] = "eval"

    pku_dict_list = pku_responses_df.to_dict(orient='records')

    with open(path+'pku_aligner_responses.json', 'w') as fp:
        json.dump(pku_dict_list, fp)
        
    aligned_by_our_aligner_df = combined_results_df[["input", "aligned_response_by_"+aligner_model_name]]
    aligned_by_our_aligner_df.columns = ["instruction", "output"]
    aligned_by_our_aligner_df["generator"] = aligner_model_name+"_aligner"
    aligned_by_our_aligner_df["dataset"] = dataset
    aligned_by_our_aligner_df["datasplit"] = "eval"

    aligned_dict_list = aligned_by_our_aligner_df.to_dict(orient='records')

    with open(path+'our_aligner_responses.json', 'w') as fp:
        json.dump(aligned_dict_list, fp)
        
    return "Done."

  
if __name__ == "__main__":
    batch_size_ = [64]
    eval_set_size_ = [15000] 
    checkpoint_ids = [2500]
    evaluator_ = ["AlpacaEval"]
    types = ["ethical", "factuality", "helpful"]
    aligner_model_names = ["gpt2", "pythia", "redpajama", "phi2"]
    base_models =  ["falcon-40b", "llama-2-13b", "llama-2-70b"]
    categories = ["squad", "individual"] 
    datasets = ["synthetic_mixed","beaverTails"]
  
    grid = list(product(datasets,categories,base_models, aligner_model_names, types, evaluator_, checkpoint_ids, eval_set_size_,batch_size_))
    
    i = int(float(sys.argv[1]))
    print("print i")
    print(sys.argv[1])
    dataset, category, base_model, aligner_model_name, type, evaluator, checkpoint_id, eval_set_size, batch_size = grid[i]    
    print("Print Aligner Model Name!")
    print(aligner_model_name)
    print("Print dataset!")
    print(dataset)

    prep_data_for_alpaca_eval(eval_set_size=eval_set_size,aligner_model_name=aligner_model_name,
                              base_model=base_model, checkpoint_id=checkpoint_id, evaluator=evaluator, batch_size=batch_size, 
                              type=type, dataset=dataset, category = category)

   


