
'''##############################################################################

NOTE: Code in this file is not being used anywhere.

'''##############################################################################


# Import modules
import os
import pandas as pd
import time
import load_data
from datasets import Dataset, DatasetDict
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

import sys, json
from itertools import product
import glob
import torch


# Prompt formatter function for when initial output is not included 
def prompt_formatter(instr, input_text):
    prompt = instr + '\n\n' + 'Input:\n' + input_text + '\n' + 'Response:'
    return prompt

# Prompt formatter function for when initial output is included 
def prompt_formatter_with_output(instr, input_text, initial_response):
    prompt = instr + '\n\n' + 'Input:\n' + input_text + '\n' + 'Response:\n' + initial_response + '\n' + 'Corrector (identify problems with response):'
    return prompt


def generate_responses(type=None, j=None, start_index=None, data_size=None, batch_size=None, model_name=None):

    modelpath = model_name
    tokenizer = AutoTokenizer.from_pretrained(modelpath, padding_side="left",device_map='auto',cache_dir='./cache/',trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(modelpath,device_map='auto',cache_dir='./cache/',trust_remote_code=True)    

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dot_id = tokenizer.encode(".")

    '''when generating, we will use the logits of right-most token to 
    predict the next token so the padding should be on the left'''
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error

    test_data_path = '../data-generation/'+type+'/data-aligner/final/'+type+'_aligner_data.csv'
    
    ds = load_data.load(test_data_path, data_size)
    test_set = ds["test"]
    test_dict = DatasetDict()
    test_dict["test"] = test_set
    test_dict.set_format(type="pandas")
    df = test_dict["test"][:]
    data = df[["input"]]
    data = data.dropna()
    
    
    corrector_principles = "Answer the following question: "
    print(corrector_principles)
    
    end_index = len(data)
    index_tracker = None
    
    start_time = time.time()

    modelname = model_name.split("/")[1]
    
    generations_path = './data/'+type+'/'+modelname+'/'
    
    bad_responses_path = './data/'+type+'/'+modelname+'/bad_responses/'
    
    final_data_path = './data/'+type+'/'+modelname+'/final/'
    
    os.makedirs(generations_path, exist_ok=True)
    os.makedirs(bad_responses_path, exist_ok=True)
    os.makedirs(final_data_path, exist_ok=True)
    
    for index in range(start_index, end_index, batch_size):
        data_batch = data[index:index+batch_size]
    
        ''' x[0] contains 'question' '''
        prompts_no_output = [prompt_formatter(corrector_principles, str(x[0]).strip()) for x in data_batch.to_numpy()]

        inputs = tokenizer(prompts_no_output, padding=True, return_tensors="pt").to(device)
    
        maxi_new_tokens = 256

        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            do_sample=False, 
            max_new_tokens = maxi_new_tokens, 
            eos_token_id = dot_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty =2.0,
        )
        generated_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        
        questions = []
        responses = []
        bad_response_indeces = []
        none_response_indeces = []
        i = 0
        for response in generated_texts:
            
            index_tracker = index + i
            if response is not None:
                result = response.replace("<|endoftext|>", '').strip()
        
                sub1_question = "\nInput:"
                sub2_question = "\nResponse:"
        
                idx1_question = result.rindex(sub1_question)
                idx2_question = result.rindex(sub2_question)
                '''Extracting question and response'''
                res_question = result[idx1_question + len(sub1_question) + 1:idx2_question].strip()
                res_response = result[idx2_question+len(sub2_question)+1:].strip()
                
                # adding extracted question and answer to their respective lists
                responses.append(res_response)

                questions.append(res_question)
                print("=====================================================================")
                print("End of response for index: ", index_tracker)
                print("=====================================================================")
                    
            else:
                none_response_indeces.append(index_tracker)
                print("*******************************************************")
                print("None response at index: ", index_tracker)
                print("*******************************************************")
    
            i = i+1
            
        batch_df = pd.DataFrame(questions, columns=['input'])
        batch_df["initial_response"] = responses
        batch_df.to_csv(generations_path+'generated_data'+str(j)+'_'+str(index_tracker)+'.csv', index=False)
       
        if len(none_response_indeces)>0:
            none_df = pd.DataFrame(none_response_indeces, columns=['none_response_index'])
            none_df.to_csv(bad_responses_path+'none'+str(j)+'.csv', index=False)
        end_time = time.time()
        print("#################################################################################################")
        print("Generated data batch saved at index: ", index_tracker, " Time taken: ", end_time-start_time)
        print("#################################################################################################")
        j = j+1

        
def gather_responses(type=None, model_name=None):
    
    final_data_path = './data/'+type+'/'+modelname+'/final/'

    os.makedirs(final_data_path, exist_ok=True)

    
    path_data = './data/'+type+'/'+modelname+'/'

    df_list = []
        
    for file_data in glob.glob(path_data + '*.csv'):
        batch_data_df = pd.read_csv(file_data)
        df_list.append(batch_data_df)
    data_df = pd.concat(df_list)
    data_df = data_df.astype(str)
    data_df.to_csv(final_data_path+type+"_initial_response_"+model_name+"_noprompt.csv", index=False)
    return data_df

if __name__ == "__main__":
    batch_size_ = [4] 
    data_size_ = [10] 
    folder_number = [0] #keeps track of folder numbers
    start_index_ = [0] #keep track of sample index
    types = ["ethical", "factuality"]

    model_names = ["mosaicml/mpt-7b", "mosaicml/mpt-7b-instruct", "tiiuae/falcon-40b", "tiiuae/falcon-40b-instruct"]

    grid = list(product(model_names, types, start_index_, folder_number, data_size_, batch_size_))
    
    i = int(float(sys.argv[1]))
    print("print i")
    print(sys.argv[1])
    model_name, type, start_index, j, data_size, batch_size = grid[i]    
    print("Print Model Name!!!")
    print(model_name)

    generate_responses(type=type, j=j, start_index=start_index, data_size=data_size, batch_size=batch_size, model_name=model_name)
    gather_responses(type=type, model_name=model_name)