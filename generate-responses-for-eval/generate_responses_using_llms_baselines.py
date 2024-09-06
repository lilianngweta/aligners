# Import modules
import os
from dotenv import load_dotenv
from genai.model import Credentials, Model
from genai.schemas import GenerateParams

import pandas as pd
import time

import sys, json
from itertools import product
import glob
import torch

#  Load Credentials
load_dotenv()
api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", None)

# Prompt formatter function for when initial output is not included 
def prompt_formatter(instr, input_text):
    prompt = instr + '\n\n' + 'Input:\n' + input_text + '\n' + 'Response:'
    return prompt

# Prompt formatter function for when initial output is included 
def prompt_formatter_with_output(instr, input_text, initial_response):
    prompt = instr + '\n\n' + 'Input:\n' + input_text + '\n' + 'Response:\n' + initial_response + '\n' + 'Corrector (identify problems with response):'
    return prompt

def gather_responses(type_=None, model_name=None):
    modelname = model_name.split("/")[1]
    final_data_path = './data_unaligned/'+type_+'/'+modelname+'/final/'
    os.makedirs(final_data_path, exist_ok=True)
    path_data = './data_unaligned/'+type_+'/'+modelname+'/'
    df_list = []   
    for file_data in glob.glob(path_data + '*.csv'):
        batch_data_df = pd.read_csv(file_data)
        df_list.append(batch_data_df)  
    data_df = pd.concat(df_list)
    data_df = data_df.astype(str)
    # save final df to final_data_path and do for all aligner types and LLMs used
    data_df.to_csv(final_data_path+type_+"_initial_response_"+modelname+"_noprompt.csv", index=False)
    return "Done combining all responses."  


def generate_responses(type_=None, j=None, start_index=None, test_size=None, data_size=None, batch_size=None, model_name=None):
    modelpath = model_name
    
    params = GenerateParams(decoding_method="greedy", min_new_tokens=1, max_new_tokens=256, stop_sequences=[".","\?","!"], repetition_penalty=2)
    
    # creds object
    creds = Credentials(api_key, api_endpoint)
    
    model = Model(modelpath, params=params, credentials=creds)    

    # load test data inputs (x)        
    data = pd.read_csv("./test_data_x/"+type_+"_test_inputx.csv")

    
    instructions = "Answer the following question: "
    print(instructions)

    end_index = len(data)
    index_tracker = None
    
    start_time = time.time()

    modelname = model_name.split("/")[1]
    
    generations_path = './data_unaligned/'+type_+'/'+modelname+'/'
    
    bad_responses_path = './data_unaligned/'+type_+'/'+modelname+'/bad_responses/'
    
    final_data_path = './data_unaligned/'+type_+'/'+modelname+'/final/'
    
    os.makedirs(generations_path, exist_ok=True)
    os.makedirs(bad_responses_path, exist_ok=True)
    os.makedirs(final_data_path, exist_ok=True)
    
    for index in range(start_index, end_index, batch_size):
        data_batch = data[index:index+batch_size]
    
        ''' x[0] contains 'question' '''
        prompts_no_output = [prompt_formatter(instructions, str(x[0]).strip()) for x in data_batch.to_numpy()]

        questions = []
        responses = []
        none_response_indeces = []
        i = 0
        for response in model.generate_async(prompts_no_output):
            
            index_tracker = index + i
            if response is not None:
                result = response.generated_text.strip()
    
                #'''Initializing substrings (used to obtain indices for string slicing)'''
                sub1_question = "\nInput:"
                sub2_question = "\nResponse:"
                
                sub_response = "\nResponse:"
    
                question = response.input_text
    
                idx1_question = question.rindex(sub1_question)
                idx2_question = question.rindex(sub2_question)
    
                res_response = result 
  
                res_question = question[idx1_question + len(sub1_question) + 1:idx2_question].strip()
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
        
    gather_responses(type_=type_, model_name=model_name)
    return "Done generating and gathering responses."



if __name__ == "__main__":
    batch_size_ = [512]
    data_size_ = [100000] 
    test_size_ = [15000] #not being used
    folder_number = [0] #keeps track of folder numbers
    start_index_ = [0] #keeps track of sample index
    types = ["synthetic_mixed", "beaverTails"]
    model_names = ["tiiuae/falcon-40b", "meta-llama/llama-2-13b-chat", "meta-llama/llama-2-70b-chat", "ibm/falcon-40b-8lang-instruct", "meta-llama/llama-2-13b", "meta-llama/llama-2-70b"]
    
    grid = list(product(model_names, types, start_index_, folder_number, test_size_, data_size_, batch_size_))
    
    i = int(float(sys.argv[1]))
    print("print i")
    print(sys.argv[1])
    model_name, type_, start_index, j, test_size, data_size, batch_size = grid[i]    
    print("Print Model Name!!!")
    print(model_name)

    generate_responses(type_=type_, j=j, start_index=start_index, test_size=test_size, data_size=data_size,
                       batch_size=batch_size, model_name=model_name)