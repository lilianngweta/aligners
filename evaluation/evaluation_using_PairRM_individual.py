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



def evaluate_with_pairRM(type=None, eval_set_size=None, aligner_model_name=None,
                             base_model=None,checkpoint_id=None, evaluator=None,batch_size=None,path_id=None,i=None):
    
    aligned_by_aligners_df = None
    if aligner_model_name == "pku_aligner":
        aligned_by_aligners_df = pd.read_csv("../generate-responses-for-eval/data_aligned/"+path_id+"/"+aligner_model_name+"/"+base_model+
                                             "/dfs/final/"+path_id+"_"+base_model+"_aligned_data_using_"+aligner_model_name+"_aligner.csv")

    else:
        aligned_by_aligners_df = pd.read_csv("../generate-responses-for-eval/data_aligned_individual/"+path_id+"/"+type+"/"+aligner_model_name+
                          "/checkpoint-"+str(checkpoint_id)+"/"+base_model+"/dfs/final/"+type+"_"+base_model+
                          "_aligned_data_using_"+aligner_model_name+"_aligner.csv")
        
        aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("</s>","")
        aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("<s>","")
        aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].apply(lambda x: str(x).strip())

        aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].str.replace("</s>","")
        aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].str.replace("<s>","")
        aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].apply(lambda x: str(x).strip())

    aligned_llm_path = "../generate-responses-for-eval/data_unaligned/"+path_id+"/"
    
    aligned_llm = [dir for dir in os.listdir(aligned_llm_path) if dir.startswith(base_model+"-")][0]

    print("Base LLM!")
    print(base_model)

    print("Aligned version of the LLM!")
    print(aligned_llm)
    
    aligned_llm_df = pd.read_csv("../generate-responses-for-eval/data_unaligned/"+path_id+"/"+aligned_llm+"/final/"+path_id+
                                   "_initial_response_"+aligned_llm+"_noprompt.csv")

    aligned_by_aligners_df = data_prep(aligned_by_aligners_df, eval_set_size)
    aligned_llm_df = data_prep(aligned_llm_df, eval_set_size)

    aligned_by_aligners_df.columns = ["input", "initial_response_"+base_model, "aligned_response_"+aligner_model_name]
    aligned_llm_df.columns = ["input", "initial_response_"+aligned_llm]

    cols = ['input']
    combined_results_df = aligned_by_aligners_df.merge(aligned_llm_df, 'outer',on=cols, indicator=True)
    combined_results_df = combined_results_df[combined_results_df['_merge'] == 'both']
    combined_results_df.drop_duplicates(subset=['input'], keep='first', inplace=True, ignore_index=True)
    combined_results_df = combined_results_df[["input", "initial_response_"+base_model, 
                                               "initial_response_"+aligned_llm, "aligned_response_"+aligner_model_name]]
    combined_results_df = combined_results_df
    
    base_inputs = combined_results_df["input"].tolist()
    aligned_inputs = combined_results_df["input"].tolist() 
    
    base_candidates_texts = combined_results_df[["initial_response_"+base_model,"aligned_response_"+aligner_model_name]].values.tolist()
    aligned_candidates_texts = combined_results_df[["initial_response_"+aligned_llm,"aligned_response_"+aligner_model_name]].values.tolist()

    base_reward_scores, base_preference_results = pair_rm(base_inputs, base_candidates_texts, batch_size)

    aligned_reward_scores, aligned_preference_results = pair_rm(aligned_inputs, aligned_candidates_texts, batch_size)
    
    class0_base_scores = base_reward_scores[:,0]
    class0_aligned_scores = aligned_reward_scores[:,0]
    class1_base_scores = base_reward_scores[:,1]
    class1_aligned_scores = aligned_reward_scores[:,1]

    class1_scores_minus_class0_base_scores = class1_base_scores - class0_base_scores
    difference_not_zero_class0_base = class1_scores_minus_class0_base_scores != 0
    class1_scores_minus_class0_base_scores = class1_scores_minus_class0_base_scores[difference_not_zero_class0_base][:eval_set_size]

    class1_scores_minus_class0_aligned_scores = class1_aligned_scores - class0_aligned_scores
    difference_not_zero_class0_aligned = class1_scores_minus_class0_aligned_scores != 0
    class1_scores_minus_class0_aligned_scores = class1_scores_minus_class0_aligned_scores[difference_not_zero_class0_aligned][:eval_set_size]

    class1_greater_than_class0_base = class1_base_scores[difference_not_zero_class0_base]>class0_base_scores[difference_not_zero_class0_base]
    accuracy_using_scores_class0_base = np.average(class1_greater_than_class0_base.astype(int)[:eval_set_size])

    class1_greater_than_class0_aligned = class1_aligned_scores[difference_not_zero_class0_aligned]>class0_aligned_scores[difference_not_zero_class0_aligned]
    accuracy_using_scores_class0_aligned = np.average(class1_greater_than_class0_aligned.astype(int)[:eval_set_size])

    print("printing accuracy!")
    print(accuracy_using_scores_class0_base)
    
    combined_results_df["class0_base_scores"]=class0_base_scores
    combined_results_df["class0_aligned_scores"]=class0_aligned_scores
    combined_results_df["class1_base_scores"]=class1_base_scores
    combined_results_df["class1_aligned_scores"]=class1_aligned_scores

    results_path = None
    if aligner_model_name == "pku_aligner":
        results_path = "./results_individual/"+path_id+"/"+aligner_model_name+"/"+base_model+"/"+evaluator+"/"
    else:
        results_path = "./results_individual/"+path_id+"/"+type+"/"+aligner_model_name+"/checkpoint-"+str(checkpoint_id)+"/"+base_model+"/"+evaluator+"/"

    os.makedirs(results_path, exist_ok=True)

    combined_results_df.to_csv(results_path+"combined_results.csv", index=False)

    # put all the results in a dictionary
    results = {}
    results['type'] = type
    results['aligner_model_name'] = aligner_model_name
    results['class0_base_scores_avg'] = float(np.average(np.mean(base_reward_scores, axis=0)[0]))
    results['class0_aligned_scores_avg'] = float(np.average(np.mean(aligned_reward_scores, axis=0)[0]))
    results['class1_base_scores'] = float(np.average(np.mean(base_reward_scores, axis=0)[1]))
    results['class1_aligned_scores'] = float(np.average(np.mean(aligned_reward_scores, axis=0)[1]))
    results['class1_scores_minus_class0_base_scores_avg'] = float(np.average(class1_scores_minus_class0_base_scores))
    results['class1_scores_minus_class0_aligned_scores_avg'] = float(np.average(class1_scores_minus_class0_aligned_scores))
    results['accuracy_class0_base'] = float(accuracy_using_scores_class0_base)
    results['accuracy_class0_aligned'] = float(accuracy_using_scores_class0_aligned)
        
    with open(results_path+'results_summary'+str(i)+'.json', 'w') as fp:
        json.dump(results, fp)
   
    return "Done."

    

if __name__ == "__main__":
    batch_size_ = [64]
    eval_set_size_ = [15000] 
    checkpoint_ids = [2500] # only used to specify the path where aligned test data is stored
    evaluator_ = ["pairRM"]
    types = ["ethical", "factuality", "helpful"]
    aligner_model_names = ["gpt2", "pythia", "redpajama", "phi2", "pku_aligner"]
    base_models =  ["falcon-40b", "llama-2-13b", "llama-2-70b"]
    path_ids = ["synthetic_mixed","beaverTails"]
  
    grid = list(product(path_ids,base_models, aligner_model_names, types, evaluator_, 
                        checkpoint_ids,eval_set_size_,batch_size_))
    
    i = int(float(sys.argv[1]))
   
    path_id, base_model, aligner_model_name, type, evaluator, checkpoint_id, eval_set_size, batch_size = grid[i]    
    print("Print Aligner Model Name!")
    print(aligner_model_name)
    print("Print path_id!")
    print(path_id)

    evaluate_with_pairRM(type=type,eval_set_size=eval_set_size,
                         aligner_model_name=aligner_model_name,base_model=base_model,
                         checkpoint_id=checkpoint_id,
                         evaluator=evaluator, batch_size=batch_size,
                         path_id=path_id,i=i)
    
    


