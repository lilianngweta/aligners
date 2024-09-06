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
                             base_model=None,checkpoint_id=None, evaluator=None,
                              batch_size=None,type=None, dataset=None, category=None):
    all_combined_dfs = []
    combined_results_df = None
    aligned_by_aligners_df = None
    if aligner_model_name == "pku_aligner":
        aligned_by_aligners_df = pd.read_csv("./data_aligned/"+dataset+"/"+aligner_model_name+"/"+base_model+"/dfs/final/"+dataset+"_"+base_model+"_aligned_data_using_"+aligner_model_name+"_aligner.csv")

        aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("</s>","")
        aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("<s>","")
        aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].apply(lambda x: str(x).strip())
            
        aligned_llm_path = "./data_unaligned/"+dataset+"/"
        
        aligned_llm = [dir for dir in os.listdir(aligned_llm_path) if dir.startswith(base_model+"-")][0]
        
        print("Base LLM!")
        print(base_model)
        
        print("Finetuned version of the LLM!")
        print(aligned_llm)
        
        aligned_llm_df = pd.read_csv("./data_unaligned/"+dataset+"/"+aligned_llm+"/final/"+dataset+
                                       "_initial_response_"+aligned_llm+"_noprompt.csv")
        
        aligned_llm_df.columns = ["input", "aligned_llm_response"]
        
        cols = ['input']
        combined_df = aligned_llm_df.merge(aligned_by_aligners_df, 'outer',on=cols, indicator=True)
        combined_df = combined_df[combined_df['_merge'] == 'both']
        combined_df.drop_duplicates(subset=['input'], keep='first', inplace=True, ignore_index=True)
        final_combined_df = combined_df[["input","initial_response", "aligned_llm_response","aligned_response"]]
        final_combined_df = final_combined_df[~(final_combined_df["aligned_response"].str.contains("Nothing"))]
        
        final_combined_df.columns = ["input", "initial_response_"+base_model, "initial_response_"+aligned_llm,
                                     "aligned_response_"+aligner_model_name]
        final_combined_df = final_combined_df.fillna('NaN')
        final_combined_df = final_combined_df.astype(str)
        combined_results_df = final_combined_df[:eval_set_size]
        
    else:
        if category == "squad":
            aligned_by_aligners_df = pd.read_csv("./data_aligned/"+dataset+"/"+aligner_model_name+
                              "/checkpoint-"+str(checkpoint_id)+"/"+base_model+"/dfs/final/"+dataset+"_"+base_model+
                              "_aligned_data_using_"+aligner_model_name+"_aligner.csv")
    
            aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("</s>","")
            aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("<s>","")
            aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].apply(lambda x: str(x).strip())
    
            aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].str.replace("</s>","")
            aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].str.replace("<s>","")
            aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].apply(lambda x: str(x).strip())
        
            
            base_df = pd.read_csv("../eval-data-generation/data_aligned/"+dataset+"/"+aligner_model_name+
                                  "/checkpoint-"+str(checkpoint_id)+"/"+base_model+"/dfs/final/"+dataset+"_"+base_model+
                                  "_aligned_data_using_"+aligner_model_name+"_aligner_all_columns.csv")
            
            
            base_df = base_df[["input_1","initial_response_og"]]
            base_df.columns = ["input","initial_response"]
            
            base_df["input"] = base_df["input"].str.replace("</s>","")
            base_df["input"] = base_df["input"].str.replace("<s>","")
            base_df["input"] = base_df["input"].apply(lambda x: str(x).strip())
            
            aligned_llm_path = "../eval-data-generation/data_unaligned/"+dataset+"/"
            
            aligned_llm = [dir for dir in os.listdir(aligned_llm_path) if dir.startswith(base_model+"-")][0]
            
            print("Base LLM!!!!")
            print(base_model)
            
            print("Finetuned version of the LLM!!!!")
            print(aligned_llm)
            
            aligned_llm_df = pd.read_csv("../eval-data-generation/data_unaligned/"+dataset+"/"+aligned_llm+"/final/"+dataset+
                                           "_initial_response_"+aligned_llm+"_noprompt.csv")
            aligned_llm_df.columns = ["input", "aligned_llm_response"]
            
            cols = ['input']
            combined_df = aligned_llm_df.merge(aligned_by_aligners_df, 'outer',on=cols, indicator=True)
            combined_df = combined_df[combined_df['_merge'] == 'both']
            combined_df.drop_duplicates(subset=['input'], keep='first', inplace=True, ignore_index=True)
            combined_df = combined_df[["input","aligned_llm_response","aligned_response"]]
            
            final_combined_df = combined_df.merge(base_df, 'outer',on=cols, indicator=True)
            final_combined_df = final_combined_df[final_combined_df['_merge'] == 'both']
            final_combined_df.drop_duplicates(subset=['input'], keep='first', inplace=True, ignore_index=True)
            final_combined_df = final_combined_df[["input","initial_response", "aligned_llm_response","aligned_response"]]
            final_combined_df = final_combined_df[~(final_combined_df["aligned_response"].str.contains("Nothing"))]
            
            final_combined_df.columns = ["input", "initial_response_"+base_model, "initial_response_"+aligned_llm,
                                         "aligned_response_"+aligner_model_name]
            combined_results_df = final_combined_df[:eval_set_size]
            
        else:
            aligned_by_aligners_df = pd.read_csv("../eval-data-generation/data_aligned_individual/"+dataset+"/"+type+"/"+aligner_model_name+
                          "/checkpoint-"+str(checkpoint_id)+"/"+base_model+"/dfs/final/"+type+"_"+base_model+
                          "_aligned_data_using_"+aligner_model_name+"_aligner.csv")
        
            aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("</s>","")
            aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].str.replace("<s>","")
            aligned_by_aligners_df["input"] = aligned_by_aligners_df["input"].apply(lambda x: str(x).strip())
    
            aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].str.replace("</s>","")
            aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].str.replace("<s>","")
            aligned_by_aligners_df["aligned_response"] = aligned_by_aligners_df["aligned_response"].apply(lambda x: str(x).strip())
    
            aligned_llm_path = "../eval-data-generation/data_unaligned/"+dataset+"/"
            
            aligned_llm = [dir for dir in os.listdir(aligned_llm_path) if dir.startswith(base_model+"-")][0]
        
            print("Base LLM!")
            print(base_model)
        
            print("Finetuned version of the LLM!")
            print(aligned_llm)
            
            aligned_llm_df = pd.read_csv("../eval-data-generation/data_unaligned/"+dataset+"/"+aligned_llm+"/final/"+dataset+
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
            
            combined_results_df = combined_results_df[:eval_set_size]

    combined_results_df = combined_results_df.fillna('NaN')
    combined_results_df = combined_results_df.astype(str)
 
    path = None
    if category == "squad":
        path = "./data_for_alpaca_eval/"+category+"/"+dataset+"/"+aligner_model_name+"/"+base_model+"/"

    else:
        path = "./data_for_alpaca_eval/"+category+"/"+dataset+"/"+type+"/"+aligner_model_name+"/"+base_model+"/"

    os.makedirs(path, exist_ok=True)
    combined_results_df.to_csv(path+"df.csv", index=False)

    '''Folders where results will be stored'''
    results_folder1 = path + "/aligned_vs_base_llm/"
    results_folder2 = path + "/aligned_vs_finetuned_llm/"
    os.makedirs(results_folder1, exist_ok=True)
    os.makedirs(results_folder2, exist_ok=True)
    
    
    combined_results_df = pd.read_csv(path+'df.csv').dropna().reset_index(drop=True)
    
    combined_results_df = combined_results_df.sample(n=800)
    base_responses_df = combined_results_df[["input", "initial_response_"+base_model]]
   
    base_responses_df.columns = ["instruction", "output"]
    base_responses_df["generator"] = base_model
    base_responses_df["dataset"] = dataset
    base_responses_df["datasplit"] = "eval"

    base_dict_list = base_responses_df.to_dict(orient='records')

    with open(path+'base_model_responses.json', 'w') as fp:
        json.dump(base_dict_list, fp)

    finetuned_df = combined_results_df[["input", "initial_response_"+aligned_llm]]
    finetuned_df.columns = ["instruction", "output"]
    finetuned_df["generator"] = aligned_llm
    finetuned_df["dataset"] = dataset
    finetuned_df["datasplit"] = "eval"
    
    finetuned_dict_list = finetuned_df.to_dict(orient='records')

    with open(path+'finetuned_model_responses.json', 'w') as fp:
        json.dump(finetuned_dict_list, fp)
        

    aligned_by_our_aligner_df = combined_results_df[["input", "aligned_response_"+aligner_model_name]]
    aligned_by_our_aligner_df.columns = ["instruction", "output"]
    aligned_by_our_aligner_df["generator"] = aligner_model_name+"_aligner"
    aligned_by_our_aligner_df["dataset"] = dataset
    aligned_by_our_aligner_df["datasplit"] = "eval"

    aligned_dict_list = aligned_by_our_aligner_df.to_dict(orient='records')

    with open(path+'aligned_responses.json', 'w') as fp:
        json.dump(aligned_dict_list, fp)

    return "Done."



if __name__ == "__main__":
    batch_size_ = [64]
    eval_set_size_ = [15000] 
    checkpoint_ids = [2500]
    evaluator_ = ["AlpacaEval"]
    types = ["ethical", "factuality", "helpful"]
    aligner_model_names = ["gpt2", "pythia", "redpajama", "phi2", "pku_aligner"]
    base_models =  ["falcon-40b", "llama-2-13b", "llama-2-70b"]
    categories = ["squad", "individual"]
    datasets = ["synthetic_mixed","beaverTails"]
  
    grid = list(product(datasets,categories,base_models, aligner_model_names, types, evaluator_, checkpoint_ids, eval_set_size_,batch_size_))
    
    i = int(float(sys.argv[1]))
    print("print i")
    print(sys.argv[1])
    dataset, category, base_model, aligner_model_name, type, evaluator, checkpoint_id, eval_set_size, batch_size = grid[i]    
    print("Print aligner model name!")
    print(aligner_model_name)
    print("Print dataset!")
    print(dataset)

    prep_data_for_alpaca_eval(eval_set_size=eval_set_size,aligner_model_name=aligner_model_name,
                              base_model=base_model, checkpoint_id=checkpoint_id, evaluator=evaluator, batch_size=batch_size, 
                              type=type, dataset=dataset, category = category)

   


