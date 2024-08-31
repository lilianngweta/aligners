import sys, json
from itertools import product
import os
import argparse

import numpy as np
import pandas as pd

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)

from utils import DataCollatorForCompletionOnlyLM

import load_data
from constants import SENTENCE_SEPARATOR, INSTRUCTION_SEPARATOR, SENTENCE_END, RESPONSE_SEPARATOR
from datasets import Dataset, DatasetDict


import torch
import gc
torch.cuda.empty_cache()
gc.collect()


os.environ['HUGGINGFACE_HUB_CACHE'] = './cache/'
os.environ['HF_HOME'] = './cache/'
os.environ['HUGGINGFACE_ASSETS_CACHE'] = './cache/'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelpath", type=str)
    parser.add_argument("--model_outdir", type=str)
    parser.add_argument("--maxlen", type=int, default=1024)
    parser.add_argument("--index", type=int)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser

parser = get_args()
args = parser.parse_args()

print("Arguments: ", args)



def load_model_and_tokenizer(modelpath):
    tokenizer = AutoTokenizer.from_pretrained(modelpath, padding_side="left", cache_dir='./cache/',trust_remote_code=True)
    model_untr = AutoModelForCausalLM.from_pretrained(modelpath, cache_dir='./cache/',trust_remote_code=True)
  
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenizer.add_special_tokens({"additional_special_tokens": [SENTENCE_SEPARATOR, INSTRUCTION_SEPARATOR, SENTENCE_END, RESPONSE_SEPARATOR]})
    
    model_untr.resize_token_embeddings(len(tokenizer))
    return model_untr, tokenizer



def dataloader(dataset_path, train_data_size, tokenizer, model, val_nsamples,maxlen,test_data_path, aligner_type):
    def tokenize_fn(examples):
        text = examples["text"]
        model_ips = tokenizer(text, max_length=maxlen,truncation=True,)
        return model_ips

    ds = load_data.load(dataset_path)

    cols = list(ds["train"].features.keys())
    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=cols)

    print('Mapped dataset: ', ds_tok)
    
    model_maxi_length = model.config.max_position_embeddings
    ds_tok = ds_tok.filter(lambda example: model_maxi_length > len(example["input_ids"]))

    response_tokid = tokenizer.encode(SENTENCE_SEPARATOR)
    response_tokid = response_tokid[0]
    ds_tok = ds_tok.filter(lambda example: response_tokid in example["input_ids"])
    print('Filtered dataset: ', ds_tok)
    
    # subsample validation set 
    if val_nsamples is not None and "validation" in ds_tok:
        val_nsamples = min(val_nsamples, len(ds_tok["validation"]))
        ds_tok["validation"] = (
            ds_tok["validation"].shuffle().select(range(val_nsamples))
        )

    if train_data_size is not None and "train" in ds_tok:
        train_data_size = min(train_data_size, len(ds_tok["train"]))
        ds_tok["train"] = (
            ds_tok["train"].select(range(train_data_size))
        )

    # Save test set to a file to be used later 
    test_set = ds["test"]
    test_dict = DatasetDict()
    test_dict["test"] = test_set
    test_dict.set_format(type="pandas")
    test_df = test_dict["test"][:]
    test_df_input = test_df[["input"]]
    test_df_input.to_csv(test_data_path+aligner_type+"_test_inputx.csv", index=False)
    
    return ds_tok




def train_aligners(val_nsamples, grad_acc_steps, nsteps, learning_rate, batch_size, train_data_size, aligner_type):
    # ---- set random seed ----#
    seed = 42
    np.random.seed(seed)
    set_seed(seed)
    
    output_path="./saved_checkpoints/"+aligner_type+"/"+args.model_outdir
    os.makedirs(output_path, exist_ok=True) # Create output directories if they don't exist

    # create a directory for storing test data if it doesn't exist already
    test_data_path="./data/"
    os.makedirs(test_data_path, exist_ok=True) 

    data_path = '../synthetic-data-generation/'+aligner_type+'/data/final/'+aligner_type+'_aligner_data.csv'
    
    # ---- load model and tokenizer ----#
    model_untr, tokenizer = load_model_and_tokenizer(args.modelpath)

    ds_tokenized = dataloader(data_path, train_data_size, tokenizer, model_untr, val_nsamples, args.maxlen,test_data_path, aligner_type)
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8, return_tensors="pt"
    )
    
    

    save_steps = 50
    eval_steps = 50
    
    '''
     Remove the use_mps_device=True argument when not running on a Mac
    ''' 
    # ---- define training arguments ---- #
    training_args = TrainingArguments(
        output_dir=output_path,
        #use_mps_device=True,
        learning_rate=learning_rate,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        weight_decay=0.01,
        save_strategy="steps",#"epoch",
        save_steps=save_steps,
        save_total_limit=None,
        max_steps=nsteps,
        warmup_steps=100,#1000,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        logging_strategy="steps",
        logging_steps=100,
        do_train=True,
        do_eval=True,
        do_predict=True,
        load_best_model_at_end=True,
        greater_is_better=False,
        evaluation_strategy="steps",#"epoch",
        eval_steps=eval_steps,
        eval_accumulation_steps=5,
        fp16=True,
        deepspeed="./ds_config_zero3.json",
    )

    
    trainer = Trainer(
        model=model_untr,
        args=training_args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=ds_tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        )


    if any(dir.startswith('checkpoint') for dir in os.listdir(output_path)):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    

    
if __name__ == "__main__":
    aligner_types = ["ethical", "factuality", "helpful"]
    train_data_size_ = [80000]
    batch_size = [1]
    learning_rate = [1e-5]
    nsteps_ = [1000000] 
    grad_acc_steps_ = [16]
    val_nsamples_ = [10000]
    
    grid = list(product(val_nsamples_, grad_acc_steps_, nsteps_, learning_rate, batch_size, train_data_size_, aligner_types))
    
    i = args.index
    
    val_nsamples, grad_acc_steps, nsteps, lr, bsz, train_data_size, aligner_typ = grid[i]
    
    print("Printing Aligner Type!")
    print(aligner_typ)
    
    train_aligners(val_nsamples=val_nsamples, grad_acc_steps=grad_acc_steps, nsteps=nsteps, learning_rate=lr, 
                   batch_size=bsz, train_data_size=train_data_size, aligner_type=aligner_typ)