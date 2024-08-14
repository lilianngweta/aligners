# Remaining code and documentation coming soon!
This repository contains code for the paper "Aligners: Decoupling LLMs and Alignment"

# 1. Synthetic Data Generation
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
