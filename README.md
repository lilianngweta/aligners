# Aligners
This repository contains code for the paper [Aligners: Decoupling LLMs and Alignment](https://arxiv.org/pdf/2403.04224)

![title](images/pipeline.png)

# 1. Synthetic data generation
## Instructions

- Navigate to the ```./synthetic-data-generation``` folder and then open either the ```ethical```, ```factuality```, or ```helpful``` folder depending on the type of dataset you are trying to generate.

- Adapt the code in ```generate_topics.py``` to the model and model source that you are going to be using. The provided code uses ```Falcon-40B``` through [IBM Foundation Models Studio](https://ibm.github.io/ibm-generative-ai/v3.0.0/getting_started.html). If you want to use ```Falcon-40B``` through Hugging Face, change the code accordingly.
  
- Run ```generate_topics.py``` using the command ```python generate_topics.py``` to generate topics.
  
- In ```deduplicate_topics.py```, provide a path to the file that contains generated topics in the ```main``` function's ```data_file``` parameter.
  
- Run ```deduplicate_topics.py``` using the command ```python deduplicate_topics.py``` to filter out invalid and duplicated topics.

- Adapt the code in ```generate_questions.py``` to the model and model source that you are going to be using. The provided code uses ```Falcon-40B``` through [IBM Foundation Models Studio](https://ibm.github.io/ibm-generative-ai/v3.0.0/getting_started.html). If you want to use ```Falcon-40B``` through Hugging Face, change the code accordingly.
  
- Run ```generate_questions.py``` using the command ```python generate_questions.py``` to generate questions (```x```).

- Create a CSV file of generated questions by running the ```json-to-df.ipynb``` jupyter notebook. The CSV file will be saved in the ```questions``` folder.

- Adapt the code in ```generate-bad-and-good-responses.ipynb``` to the model and model source that you are going to be use. The provided code uses ```Falcon-40B``` through [IBM Foundation Models Studio](https://ibm.github.io/ibm-generative-ai/v3.0.0/getting_started.html). If you want to use ```Falcon-40B``` through Hugging Face, change the code accordingly.

- Generate misaligned (```y```) and aligned (```y'```) responses to every question (```x```) by running the ```generate-bad-and-good-responses.ipynb``` jupyter notebook.

- Clean data for inspector and aligner training by running the ```clean-data-for-inspector-and-aligner-training.ipynb``` notebook to filter out bad samples.


# 2. Train inspectors and aligners
## Instructions

After generating synthetic data in step 1, train inspectors and aligners as follows:

- To train inspectors, navigate to the ```./inspector-training``` folder and run the ```train_inspectors.sh``` bash script.
  
- To train aligners, navigate to the ```./aligner-training``` folder and train GPT-2 Large, Pythia-1.4B, RedPajama-3B, and Phi-2 aligners by running ```train_aligners_gpt2.sh```, ```train_aligners_pythia.sh```, ```train_aligners_redpajama.sh```, and ```train_aligners_phi2.sh``` bash scripts, respectively.

NOTE: Adapt the bash scripts to the system/cluster that you are using to specify the number of nodes, GPUs, etc. Example bash scripts for when you are running on a cluster that uses a Slurm job scheduler are in the ```./aligner-training``` folder.


# 3. Generate responses for evaluation using aligners squad and baseline models
## Instructions

Instructions coming soon, ignore the text below ... <br><br>

Create ```synthetic_mixed``` made of ethical, factuality, and helful test sets questions ```(x)``` (5000 samples each) and put it in ```./test_data_x``` by following the naming convention ```{data_name}_test_inputx.csv``` as seen in example test data files..<br><br>.

Run the Jupyter notebook ```beaver_tails_data_prep.ipynb``` to download the BeaverTails evaluation dataset from Hugging Face, extract questions ```(x)``` from it, and save it as a csv file in ```./test_data_x```.



<br><br>

Unaligned responses generated by LLMs that we use as baselines are saved in ```./data_unaligned```

Provide a path to saved inspector checkpoints in the ```generate_responses``` function in ```generate_using_aligners_squad.py```. <br>

Aligned data is stored in ```./data_aligned``` for aligners squad and Ji et al's aligner and ```./data_aligned_individual``` for individual aligners.<br>


AlpacaEval data prep...prepped data is saved in ```./data_for_alpaca_eval```


