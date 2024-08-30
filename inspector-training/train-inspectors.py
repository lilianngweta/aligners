import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_metric
import sys, json
from itertools import product
import os
import argparse

import evaluate
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from constants import RESPONSE_SEPARATOR


def train_inspector(inspector_type=""):
    output_path="./inspector_checkpoints/bert/"+inspector_type
    # Create output directories if they don't exist
    os.makedirs(output_path, exist_ok=True)

    data_df = pd.read_csv('../synthetic-data-generation/'+inspector_type+'/data/final/'+inspector_type+'_inspector_data.csv', lineterminator='\n')
    data_df = data_df.sample(frac=1).reset_index(drop=True) #shuffle data
    data_df = data_df.dropna()
    data_df

    # convert dataframe to a hugging face dataset object
    ds = Dataset.from_pandas(data_df)

    train_test_split = ds.train_test_split(test_size=0.2)
    val_test_split = train_test_split["test"].train_test_split(test_size=0.5)

    train_data = train_test_split["train"]
    eval_data = val_test_split["train"]
    test_data = val_test_split["test"]
    data = DatasetDict()
    data["train"] = train_test_split["train"]
    data["validation"] = val_test_split["train"]
    data["test"] = val_test_split["test"]
 

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenizer.add_special_tokens({"additional_special_tokens": [RESPONSE_SEPARATOR]})
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    tokenized_data = data.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.resize_token_embeddings(len(tokenizer))


    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        print(predictions)
        return metric.compute(predictions=predictions, references=labels)

    '''
     Remove the use_mps_device=True argument when not running on a Mac
    ''' 
    training_args = TrainingArguments(
        output_dir= output_path,
    #     use_mps_device=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=40,
        weight_decay=0.01,
        save_strategy="epoch",
        report_to="tensorboard",
        do_train=True,
        do_eval=True,
        do_predict=True,
        load_best_model_at_end=True,
        metric_for_best_model = "accuracy",
        greater_is_better=True,
        evaluation_strategy="epoch",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()


if __name__ == "__main__":
    inspector_types = ["ethical", "factuality", "helpful"]
    grid = list(product(inspector_types))
    
    i = int(float(sys.argv[1]))
    inspector_typ = grid[i]
    
    inspector_typ = inspector_typ[0]
    
    print("Printing Inspector Type!")
    print(inspector_typ)
    
    train_inspector(inspector_type=inspector_typ)


