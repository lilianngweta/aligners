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

from constants import SENTENCE_SEPARATOR, INSTRUCTION_SEPARATOR, RESPONSE_SEPARATOR, SENTENCE_END
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


import torch
import gc
torch.cuda.empty_cache()
gc.collect()


os.environ['HUGGINGFACE_HUB_CACHE'] = './cache/'
os.environ['HF_HOME'] = './cache/'
os.environ['HUGGINGFACE_ASSETS_CACHE'] = './cache/'

    

def data_prep(df, size,type):
    size = min(size, len(df))
    df = df.fillna('NaN')
    df = df.astype(str)
    df = df[:size]
    df = df.sort_values(by=['input','initial_response'])    
    texts = list(df["input"] + INSTRUCTION_SEPARATOR + df["initial_response"]+SENTENCE_SEPARATOR)
    class0_for_inspector = list(df["input"] + RESPONSE_SEPARATOR + df["initial_response"])
    return class0_for_inspector, texts 



def inspector_data_prep(questions, responses):
    df = pd.DataFrame(questions, columns=['input'])
    df["response"] = responses
    data_for_inspector = list(df["input"] + RESPONSE_SEPARATOR + df["response"])
    return data_for_inspector 


def aligner_data_prep(questions, responses):
    df = pd.DataFrame(questions, columns=['input'])
    df["response"] = responses
    data_for_aligner = list(df["input"] + INSTRUCTION_SEPARATOR + df["response"]+SENTENCE_SEPARATOR)
    return data_for_aligner 



def inspect_responses(text_list, inspector_tokenizer, inspector_model):
    df = pd.DataFrame(text_list, columns = ["text"])
    df = df.fillna('NaN')
    df = df.astype(str)
    df_ = Dataset.from_pandas(df)
    df_dict = DatasetDict()
    df_dict["test"] = df_
    
    def preprocess_function(examples):
        return inspector_tokenizer(examples["text"], truncation=True, max_length=512)
    
    test_tokenized_data = df_dict.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=inspector_tokenizer)
    test_args = TrainingArguments(output_dir="./inspector_results",)
    # Define test trainer
    test_trainer = Trainer(model=inspector_model, args=test_args, tokenizer=inspector_tokenizer, data_collator=data_collator,)
    # Make prediction
    raw_pred, labels,metrics = test_trainer.predict(test_tokenized_data["test"])
    raw_pred_torch = torch.from_numpy(raw_pred)
    # turn logits into probabilities
    probs = torch.nn.Softmax()(raw_pred_torch) 
    # Get probability corresponding to predicting y'(1)
    class1_probs = probs[:,1].numpy()
    return class1_probs

def format_no_aligner_applied_data(texts):
    class0_texts = []
    class1_texts = []
    questions = []
    init_sep = INSTRUCTION_SEPARATOR.replace(" ", '').replace("[", '').replace("]", '')
    aligned_sep = SENTENCE_SEPARATOR.replace(" ", '').replace("[", '').replace("]", '')
    for text in texts:
        split_text = re.split(' \['+init_sep+'\] | \['+aligned_sep+'\] ', text)
        question = split_text[0]
        class0_text = split_text[1]
        class1_text = split_text[1]
        questions.append(question)
        class0_texts.append(class0_text)
        class1_texts.append(class1_text)
    return questions, class0_texts, class1_texts


def compute_scores_and_determine_aligner_application_order(inspector_data_batch,factuality_inspector_tokenizer,
                                                           factuality_inspector_model, ethical_inspector_tokenizer,
                                                           ethical_inspector_model,helpful_inspector_tokenizer,
                                                           helpful_inspector_model):
    aligner_types = np.array(["factuality", "ethical", "helpful"])
    
    factuality_inspector_scores = inspect_responses(inspector_data_batch, factuality_inspector_tokenizer, factuality_inspector_model)
    ethical_inspector_scores = inspect_responses(inspector_data_batch, ethical_inspector_tokenizer, ethical_inspector_model) 
    
    helpful_inspector_scores = inspect_responses(inspector_data_batch, helpful_inspector_tokenizer, helpful_inspector_model)

    factuality_inspector_scores_avg = np.average(factuality_inspector_scores)
    ethical_inspector_scores_avg = np.average(ethical_inspector_scores)
    helpful_inspector_scores_avg = np.average(helpful_inspector_scores)

    batch_average_scores_of_inspectors = np.array([factuality_inspector_scores_avg, 
                                                   ethical_inspector_scores_avg, helpful_inspector_scores_avg])
    sorted_indeces = np.argsort(batch_average_scores_of_inspectors)
    aligners_order = list(aligner_types[sorted_indeces])

    return factuality_inspector_scores, ethical_inspector_scores, helpful_inspector_scores, aligners_order



def apply_aligner(inspector_scores,inspector_score_threshold,texts_batch,aligner_model,aligner_tokenizer):
    inspector_scores_less_than_threshold = inspector_scores < inspector_score_threshold
    no_aligner_needed = np.array(texts_batch)[~inspector_scores_less_than_threshold]
    aligner_needed = np.array(texts_batch)[inspector_scores_less_than_threshold]
    
    questions_batch_no_aligner, class0_batch_no_aligner, class1_batch_no_aligner = format_no_aligner_applied_data(no_aligner_needed)
    
    questions_batch_aligned, class0_batch_aligned, class1_batch_aligned = batch_responses_for_eval(aligner_model, 
                                                                                                   aligner_tokenizer,list(aligner_needed))
    all_questions = questions_batch_no_aligner + questions_batch_aligned
    all_class0 = class0_batch_no_aligner + class0_batch_aligned
    all_class1 = class1_batch_no_aligner + class1_batch_aligned
    return all_questions, all_class0, all_class1



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

    if len(texts_batch) != 0:
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
                end = SENTENCE_END.replace(" ", '').replace("[", '').replace("]", '')
                class1_text_ = split_gen_text[2]
                if SENTENCE_END in class1_text_:  
                    class1_text = re.split(' \['+end+'\]', class1_text_)[0]
                else:
                    class1_text = class1_text_
            elif len(split_gen_text)==2:
                question = split_gen_text[0]
                class0_text = split_gen_text[1]
                class1_text = " "
            questions.append(question)
            class0_texts.append(class0_text)
            class1_texts.append(class1_text)
    return questions, class0_texts, class1_texts
    



def gather_responses(type,aligner_model_name,checkpoint_id,base_model,aligned_dfs_path):
    final_aligned_df_path = aligned_dfs_path+"final/"
    
    os.makedirs(final_aligned_df_path, exist_ok=True)
    df_list = []   
    for file_data in glob.glob(aligned_dfs_path + '*.csv'):
        batch_data_df = pd.read_csv(file_data)
        df_list.append(batch_data_df)  
    data_df = pd.concat(df_list)
    data_df = data_df.astype(str)
    data_df["input_1"] = data_df["input_1"].str.replace("</s>","")
    data_df["input_2"] = data_df["input_2"].str.replace("</s>","")
    data_df["input_3"] = data_df["input_3"].str.replace("</s>","")
    data_df["input_1"] = data_df["input_1"].str.replace("<s>","")
    data_df["input_2"] = data_df["input_2"].str.replace("<s>","")
    data_df["input_3"] = data_df["input_3"].str.replace("<s>","")

    data_df["aligned_response_1"] = data_df["aligned_response_1"].str.replace("</s>","")
    data_df["aligned_response_2"] = data_df["aligned_response_2"].str.replace("</s>","")
    data_df["aligned_response_3"] = data_df["aligned_response_3"].str.replace("</s>","")
    data_df["aligned_response_1"] = data_df["aligned_response_1"].str.replace("<s>","")
    data_df["aligned_response_2"] = data_df["aligned_response_2"].str.replace("<s>","")
    data_df["aligned_response_3"] = data_df["aligned_response_3"].str.replace("<s>","")
    data_df.to_csv(final_aligned_df_path+type+"_"+base_model+"_aligned_data_using_"+aligner_model_name+"_aligner_all_columns.csv", index=False)
    data_df = data_df[["input_3", "aligned_response_3"]]
    data_df.columns = ["input", "aligned_response"]
    # save final df to final_aligned_df_path and do for all aligner types
    data_df.to_csv(final_aligned_df_path+type+"_"+base_model+"_aligned_data_using_"+aligner_model_name+"_aligner.csv", index=False)
    return "Done combining all responses."



def generate_responses(batch_size,aligned_dfs_path,checkpoint_id,type,aligner_model_name,
                       base_model,unaligned_df,eval_set_size,inspector_score_threshold):
    ethical_modelpath = "../aligner-training/saved_checkpoints/ethical/"+aligner_model_name+"/checkpoint-"+str(checkpoint_id) 
    ethical_tokenizer = AutoTokenizer.from_pretrained(ethical_modelpath, padding_side="left",device_map='auto')
    ethical_model = AutoModelForCausalLM.from_pretrained(ethical_modelpath,device_map='auto')
    
    factuality_modelpath = "../aligner-training/saved_checkpoints/factuality/"+aligner_model_name+"/checkpoint-"+str(checkpoint_id)
    factuality_tokenizer = AutoTokenizer.from_pretrained(factuality_modelpath, padding_side="left",device_map='auto')
    factuality_model = AutoModelForCausalLM.from_pretrained(factuality_modelpath,device_map='auto')
    
    helpful_modelpath = "../aligner-training/saved_checkpoints/helpful/"+aligner_model_name+"/checkpoint-"+str(checkpoint_id) 
    helpful_tokenizer = AutoTokenizer.from_pretrained(helpful_modelpath, padding_side="left",device_map='auto')
    helpful_model = AutoModelForCausalLM.from_pretrained(helpful_modelpath,device_map='auto')
    
    factuality_inspector_model_path = "../inspector-training/inspector_checkpoints/bert/factuality/checkpoint-4668"
    factuality_inspector_model = AutoModelForSequenceClassification.from_pretrained(factuality_inspector_model_path, num_labels=2)
    factuality_inspector_tokenizer = AutoTokenizer.from_pretrained(factuality_inspector_model_path)
    
    ethical_inspector_model_path = "../inspector-training/inspector_checkpoints/bert/ethical/checkpoint-4668"
    ethical_inspector_model = AutoModelForSequenceClassification.from_pretrained(ethical_inspector_model_path, num_labels=2)
    ethical_inspector_tokenizer = AutoTokenizer.from_pretrained(ethical_inspector_model_path)
    
    helpful_inspector_model_path = "../inspector-training/inspector_checkpoints/bert/helpful/checkpoint-4668"
    helpful_inspector_model = AutoModelForSequenceClassification.from_pretrained(helpful_inspector_model_path, num_labels=2)
    helpful_inspector_tokenizer = AutoTokenizer.from_pretrained(helpful_inspector_model_path)

    class0_for_inspector, texts = data_prep(unaligned_df, eval_set_size,type)

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
        texts_batch = texts[index : index + batch_size]
        class0_for_inspector_batch = class0_for_inspector[index : index + batch_size]
    
        factuality_inspector_scores, ethical_inspector_scores, helpful_inspector_scores, aligners_order = compute_scores_and_determine_aligner_application_order(
            class0_for_inspector_batch,factuality_inspector_tokenizer,factuality_inspector_model,ethical_inspector_tokenizer,
            ethical_inspector_model,helpful_inspector_tokenizer,helpful_inspector_model)

        # Using inspectors and aligners squad to align responses #
        ###########################################################################################################
        first_aligner = aligners_order[0]
        print("Aligner being applied")
        print(first_aligner)
        
        inspector_scores = None
        aligner_model = None
        aligner_tokenizer = None
    
        if first_aligner == "ethical":
            inspector_scores = ethical_inspector_scores
            aligner_model = ethical_model
            aligner_tokenizer = ethical_tokenizer
    
        elif first_aligner == "factuality":
            inspector_scores = factuality_inspector_scores
            aligner_model = factuality_model
            aligner_tokenizer = factuality_tokenizer
    
        else:
            inspector_scores = helpful_inspector_scores
            aligner_model = helpful_model
            aligner_tokenizer = helpful_tokenizer
        
        all_questions1, all_class0_1, all_class1_1 = apply_aligner(inspector_scores,inspector_score_threshold,texts_batch,aligner_model,aligner_tokenizer)
        
        class0_for_inspector1 = inspector_data_prep(all_questions1, all_class0_1)
        factuality_inspector_scores_before1, ethical_inspector_scores_before1, helpful_inspector_scores_before1, _ = compute_scores_and_determine_aligner_application_order(
            class0_for_inspector1,factuality_inspector_tokenizer,factuality_inspector_model,ethical_inspector_tokenizer,
            ethical_inspector_model,helpful_inspector_tokenizer,helpful_inspector_model)
        
        class1_for_inspector1 = inspector_data_prep(all_questions1, all_class1_1)
        factuality_inspector_scores_after1, ethical_inspector_scores_after1, helpful_inspector_scores_after1, _ = compute_scores_and_determine_aligner_application_order(
            class1_for_inspector1,factuality_inspector_tokenizer,factuality_inspector_model,ethical_inspector_tokenizer,
            ethical_inspector_model,helpful_inspector_tokenizer,helpful_inspector_model)
    
        data_df = pd.DataFrame(all_questions1, columns=["input_1"])
        data_df["initial_response_og"] = all_class0_1
        data_df["aligned_response_1"] = all_class1_1
        data_df["factuality_inspector_scores_before_1"] = factuality_inspector_scores_before1
        data_df["factuality_inspector_scores_after_1"] = factuality_inspector_scores_after1
        
        data_df["ethical_inspector_scores_before_1"] = ethical_inspector_scores_before1
        data_df["ethical_inspector_scores_after_1"] = ethical_inspector_scores_after1
        
        data_df["helpful_inspector_scores_before_1"] = helpful_inspector_scores_before1
        data_df["helpful_inspector_scores_after_1"] = helpful_inspector_scores_after1
        data_df["aligners_applied_1"] = first_aligner

    
       ##########################################################################################################
        
        second_aligner = aligners_order[1]
        print("Aligner being applied")
        print(second_aligner)
    
        inspector_scores = None
        aligner_model = None
        aligner_tokenizer = None
    
        if second_aligner == "ethical":
            inspector_scores = ethical_inspector_scores_after1
            aligner_model = ethical_model
            aligner_tokenizer = ethical_tokenizer
    
        elif second_aligner == "factuality":
            inspector_scores = factuality_inspector_scores_after1
            aligner_model = factuality_model
            aligner_tokenizer = factuality_tokenizer
    
        else:
            inspector_scores = helpful_inspector_scores_after1
            aligner_model = helpful_model
            aligner_tokenizer = helpful_tokenizer
    
        data_for_aligner2 = aligner_data_prep(all_questions1, all_class1_1)
        all_questions2, all_class0_2, all_class1_2 = apply_aligner(inspector_scores,inspector_score_threshold,data_for_aligner2,aligner_model,aligner_tokenizer)
        
        class0_for_inspector2 = inspector_data_prep(all_questions2, all_class0_2)
        factuality_inspector_scores_before2, ethical_inspector_scores_before2, helpful_inspector_scores_before2, _ = compute_scores_and_determine_aligner_application_order(
            class0_for_inspector2,factuality_inspector_tokenizer,factuality_inspector_model,ethical_inspector_tokenizer,
            ethical_inspector_model,helpful_inspector_tokenizer,helpful_inspector_model)
        
        class1_for_inspector2 = inspector_data_prep(all_questions2, all_class1_2)
        factuality_inspector_scores_after2, ethical_inspector_scores_after2, helpful_inspector_scores_after2, _ = compute_scores_and_determine_aligner_application_order(
            class1_for_inspector2,factuality_inspector_tokenizer,factuality_inspector_model,ethical_inspector_tokenizer,
            ethical_inspector_model,helpful_inspector_tokenizer,helpful_inspector_model)
    
        data_df["input_2"] = all_questions2
        data_df["initial_response_2"] = all_class0_2
        data_df["aligned_response_2"] = all_class1_2
        data_df["factuality_inspector_scores_before_2"] = factuality_inspector_scores_before2
        data_df["factuality_inspector_scores_after_2"] = factuality_inspector_scores_after2
        
        data_df["ethical_inspector_scores_before_2"] = ethical_inspector_scores_before2
        data_df["ethical_inspector_scores_after_2"] = ethical_inspector_scores_after2
        
        data_df["helpful_inspector_scores_before_2"] = helpful_inspector_scores_before2
        data_df["helpful_inspector_scores_after_2"] = helpful_inspector_scores_after2
        data_df["aligners_applied_2"] = first_aligner+","+second_aligner
    
    
        ##########################################################################################################
        
        third_aligner = aligners_order[2]
        print("Aligner being applied")
        print(third_aligner)
    
        inspector_scores = None
        aligner_model = None
        aligner_tokenizer = None
    
        if third_aligner == "ethical":
            inspector_scores = ethical_inspector_scores_after2
            aligner_model = ethical_model
            aligner_tokenizer = ethical_tokenizer
    
        elif third_aligner == "factuality":
            inspector_scores = factuality_inspector_scores_after2
            aligner_model = factuality_model
            aligner_tokenizer = factuality_tokenizer
    
        else:
            inspector_scores = helpful_inspector_scores_after2
            aligner_model = helpful_model
            aligner_tokenizer = helpful_tokenizer
    
        data_for_aligner3 = aligner_data_prep(all_questions2, all_class1_2)
        all_questions3, all_class0_3, all_class1_3 = apply_aligner(inspector_scores,inspector_score_threshold,data_for_aligner3,aligner_model,aligner_tokenizer)
        
        class0_for_inspector3 = inspector_data_prep(all_questions3, all_class0_3)
        factuality_inspector_scores_before3, ethical_inspector_scores_before3, helpful_inspector_scores_before3, _ = compute_scores_and_determine_aligner_application_order(
            class0_for_inspector3,factuality_inspector_tokenizer,factuality_inspector_model,ethical_inspector_tokenizer,
            ethical_inspector_model,helpful_inspector_tokenizer,helpful_inspector_model)
        
        class1_for_inspector3 = inspector_data_prep(all_questions3, all_class1_3)
        factuality_inspector_scores_after3, ethical_inspector_scores_after3, helpful_inspector_scores_after3, _ = compute_scores_and_determine_aligner_application_order(
            class1_for_inspector3,factuality_inspector_tokenizer,factuality_inspector_model,ethical_inspector_tokenizer,
            ethical_inspector_model,helpful_inspector_tokenizer,helpful_inspector_model)
    
        data_df["input_3"] = all_questions3
        data_df["initial_response_3"] = all_class0_3
        data_df["aligned_response_3"] = all_class1_3
        data_df["factuality_inspector_scores_before_3"] = factuality_inspector_scores_before3
        data_df["factuality_inspector_scores_after_3"] = factuality_inspector_scores_after3
        
        data_df["ethical_inspector_scores_before_3"] = ethical_inspector_scores_before3
        data_df["ethical_inspector_scores_after_3"] = ethical_inspector_scores_after3
        
        data_df["helpful_inspector_scores_before_3"] = helpful_inspector_scores_before3
        data_df["helpful_inspector_scores_after_3"] = helpful_inspector_scores_after3
        data_df["aligners_applied_3"] = first_aligner+","+second_aligner+","+third_aligner
    
        data_df.to_csv(aligned_dfs_path+"df_checkpoint-"+str(checkpoint_id)+"_index_"+str(index)+".csv", index=False)
        print("At index ", index)   

    if any(dir.startswith('final') for dir in os.listdir(aligned_dfs_path)):
        print("Completed!")
    else:
        gather_responses(type,aligner_model_name,checkpoint_id,base_model,aligned_dfs_path)
    return "DONE."
             


def generate_responses_using_aligners(type=None, batch_size=None, eval_set_size=None,aligner_model_name=None,
                                      base_model=None,checkpoint_id=None,inspector_score_threshold=None):
    unaligned_df = None
    if base_model=="qualitative":
        unaligned_df = pd.read_csv("./qualitative_data/qualitative_eval_data.csv") 

    else:
        unaligned_df = pd.read_csv("./data_unaligned/"+type+"/"+base_model+"/final/"+type+"_initial_response_"+base_model+"_noprompt.csv")
        
    aligned_dfs_path = "./data_aligned/"+type+"/"+aligner_model_name+"/checkpoint-"+str(checkpoint_id)+"/"+base_model+"/dfs/"
    os.makedirs(aligned_dfs_path, exist_ok=True)
    
    modelpath = "../aligner-training/saved_checkpoints/"+type+"/"+aligner_model_name+"/checkpoint-"+str(checkpoint_id)  
    
    '''Generate responses using aligners'''
    generate_responses(batch_size,aligned_dfs_path,checkpoint_id,type,aligner_model_name,base_model,
                       unaligned_df,eval_set_size,inspector_score_threshold)
    
    
    

if __name__ == "__main__":
    batch_size_ = [64]
    eval_set_size_ = [15000] 
    checkpoint_ids = [2500]
    inspector_score_threshold_ = [0.5]
    types = ["synthetic_mixed", "beaverTails"]
    aligner_model_names = ["gpt2", "pythia", "redpajama", "phi2"]
    base_models =  ["falcon-40b", "llama-2-13b", "llama-2-70b"]

    grid = list(product(base_models, aligner_model_names, types, inspector_score_threshold_, 
                        checkpoint_ids, eval_set_size_, batch_size_))
    
    i = int(float(sys.argv[1]))
    print("print i")
    print(sys.argv[1])
    base_model, aligner_model_name, type, inspector_score_threshold, checkpoint_id, eval_set_size, batch_size = grid[i]    
    print("Print Aligner Model Name!")
    print(aligner_model_name)
    print("Print Base Model Name!")
    print(base_model)

    generate_responses_using_aligners(type=type, batch_size=batch_size, 
                                      eval_set_size=eval_set_size, aligner_model_name=aligner_model_name,
                                      base_model=base_model,checkpoint_id=checkpoint_id, 
                                      inspector_score_threshold=inspector_score_threshold)



