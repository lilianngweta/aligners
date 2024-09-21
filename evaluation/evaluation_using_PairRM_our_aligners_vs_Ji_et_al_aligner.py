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
)

from constants import SENTENCE_SEPARATOR, INSTRUCTION_SEPARATOR, RESPONSE_SEPARATOR,SENTENCE_END
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
import llm_blender


import torch
import gc
torch.cuda.empty_cache()
gc.collect()


os.environ['HUGGINGFACE_HUB_CACHE'] = './cache/'
os.environ['HF_HOME'] = './cache/'
os.environ['HUGGINGFACE_ASSETS_CACHE'] = './cache/'


def pair_rm(inputs, candidates_texts, batch_size):
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")

    reward_scores = []
    preference_results = []
    for index in range(0, len(inputs), batch_size):
        inputs_batch = inputs[index: index+batch_size]
        candidates_texts_batch = candidates_texts[index: index+batch_size]
        
        rank_scores = blender.rank(inputs_batch, candidates_texts_batch, return_scores=True, batch_size=batch_size)
        reward_scores.append(rank_scores)
        
        candidates_A = [cands[0] for cands in candidates_texts_batch]
        candidates_B = [cands[1] for cands in candidates_texts_batch]
    
        comparison_results = blender.compare(inputs_batch, candidates_A, candidates_B)
        preference_results.append(comparison_results)
        
        print("At index: ", index )
        
    reward_scores = np.concatenate(reward_scores)
    preference_results = np.concatenate(preference_results)
    return reward_scores, preference_results



def evaluate_with_pairRM(eval_set_size=None, aligner_model_name=None,
                             base_model=None,checkpoint_id=None, evaluator=None,batch_size=None,type=None,i=None):
    all_combined_dfs = []
    all_types = [type]
    for type in all_types:
        
        aligned_by_aligners_df = pd.read_csv("../generate-responses-for-eval/data_aligned/"+type+"/"+aligner_model_name+
                          "/checkpoint-"+str(checkpoint_id)+"/"+base_model+"/dfs/final/"+type+"_"+base_model+
                          "_aligned_data_using_"+aligner_model_name+"_aligner.csv")

        aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("</s>","")
        aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("<s>","")
        aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].apply(lambda x: str(x).strip())

        aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].str.replace("</s>","")
        aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].str.replace("<s>","")
        aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].apply(lambda x: str(x).strip())
        
        aligned_by_aligners_df.columns = ["input", "aligned_response_by_our_aligner"]
    
    
        aligned_by_pku_aligner_df = pd.read_csv("../generate-responses-for-eval/data_aligned/"+type+"/pku_aligner/"+base_model+
                                                 "/dfs/final/"+type+"_"+base_model+"_aligned_data_using_pku_aligner_aligner.csv")
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
        final_combined_df = final_combined_df[:eval_set_size]
        
        all_combined_dfs.append(final_combined_df)   
    
    combined_results_df = pd.concat(all_combined_dfs)
    
    inputs = combined_results_df["input"].tolist()
    
    candidates_texts = combined_results_df[["aligned_response_by_PKU_aligner","aligned_response_by_"+aligner_model_name]].values.tolist()

    reward_scores, preference_results = pair_rm(inputs, candidates_texts, batch_size)
    
    class0_scores = reward_scores[:,0]
    class1_scores = reward_scores[:,1]

    class1_scores_minus_class0_scores = class1_scores - class0_scores
    difference_not_zero = class1_scores_minus_class0_scores != 0
    class1_scores_minus_class0_scores = class1_scores_minus_class0_scores[difference_not_zero]

    accuracy = 1 - np.average(preference_results.astype(int))

    class1_greater_than_class0 = class1_scores[difference_not_zero]>class0_scores[difference_not_zero]
    accuracy_using_scores = np.average(class1_greater_than_class0.astype(int))

    print("printing accuracy!")
    print(accuracy_using_scores)
    
    combined_results_df["class0_scores"]=class0_scores
    combined_results_df["class1_scores"]=class1_scores

    results_path = "./results/all_datasets/pku_vs_our_aligners/"+type+"/"+aligner_model_name+"/checkpoint-"+str(checkpoint_id)+"/"+base_model+"/"+evaluator+"/"
    os.makedirs(results_path, exist_ok=True)

    combined_results_df.to_csv(results_path+"combined_results.csv", index=False)  

    # put all the results in a dictionary
    results = {}
    results['base_model'] = base_model
    results['evaluator'] = evaluator
    results['aligner_model_name_being_compared_with_pku_aligner'] = aligner_model_name
    results['class0_scores_avg'] = float(np.average(np.mean(reward_scores, axis=0)[0]))
    results['class1_scores_avg'] = float(np.average(np.mean(reward_scores, axis=0)[1]))
    results['class1_scores_minus_class0_scores_avg'] = float(np.average(class1_scores_minus_class0_scores))
    results['accuracy_no_drop_of_same_scores'] = float(accuracy)
    results['accuracy_using_scores_same_scores_dropped'] = float(accuracy_using_scores)
        
    with open(results_path+'results_summary'+str(i)+'.json', 'w') as fp:
        json.dump(results, fp)
   
    return "Done."



if __name__ == "__main__":
    batch_size_ = [32]
    eval_set_size_ = [15000] 
    checkpoint_ids = [2500] # only used to specify the path where aligned test data is stored
    evaluator_ = ["pairRM"]
    types = ["synthetic_mixed", "beaverTails"]
    aligner_model_names = ["gpt2", "pythia", "redpajama", "phi2"]
    base_models = ["falcon-40b", "llama-2-13b", "llama-2-70b"]

    grid = list(product(base_models, aligner_model_names, types, evaluator_, checkpoint_ids, eval_set_size_,batch_size_))
    
    i = int(float(sys.argv[1]))
    
    base_model, aligner_model_name, type, evaluator, checkpoint_id, eval_set_size, batch_size = grid[i]  
    print("Print Aligner Model Name!")
    print(aligner_model_name)
    print("Print base_model!")
    print(base_model)
    print("Print type!")
    print(type)
    
    evaluate_with_pairRM(eval_set_size=eval_set_size,
                                       aligner_model_name=aligner_model_name,base_model=base_model,
                                       checkpoint_id=checkpoint_id, evaluator=evaluator, batch_size=batch_size, type=type, i=i)
    
    


