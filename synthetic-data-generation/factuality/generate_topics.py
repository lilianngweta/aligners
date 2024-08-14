'''
The code in this file was adapted from Sun et al.'s Topic-Guided Red-Teaming Self-Instruct code found in  https://github.com/IBM/Dromedary/blob/dromedary_v1/training/step1_topic_guided_red_teaming_self_instruct/generate_tgrt_topic.py 
'''



"""Topic generation for 20 question types."""

# Import modules
import os
from dotenv import load_dotenv
from genai.model import Credentials, Model
from genai.schemas import GenerateParams, ModelType

import pandas as pd
import time

import json
import os
import sys
import random
import string

import tqdm
import fire


'''Setup the model to use for generating topics'''
#  Load Credentials
load_dotenv()
api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", None)


params = GenerateParams(
    decoding_method="sample",
    max_new_tokens=300,
    min_new_tokens=1,
    stream=False,
    temperature=1.0,
    top_p=1,
)

generate_params = GenerateParams(decoding_method="sample", max_new_tokens=300, min_new_tokens=1,
                                 top_p=0.9, stream=False)

# creds object
creds = Credentials(api_key, api_endpoint)

# model object
model = Model("tiiuae/falcon-40b", params=params, credentials=creds)





def brainstorm_topics(
    model,
    batch_prompts,
    meta_prompt,
    generate_max_len,
):
    question_types = []
    all_topics = []

    for prompt in batch_prompts:
        topics = prompt["topics"]
        question_type = prompt["question_type"]
        question_types.append(question_type.strip())
        all_topics.append(topics)
    all_results = []
    prompts = []
    for question_type, topics in zip(question_types, all_topics):
        prompt = meta_prompt.format(question_type) + "\n\n"
        for i, topic in enumerate(topics):
            prompt += f"{i+1}. {topic}\n"
        prompts.append(prompt)


    results = model.generate_async(prompts, ordered = True)
    
    for question_type, result in zip(question_types, results):
        if result is not None:
            result = result.generated_text
            single_results = result.split("\n")[:20]
            formatted_single_results = []
            for single_result in single_results:
                try:
                    single_result = single_result.split(". ", 1)[-1]
                    if len(single_result.split()) > 3:
                        raise Exception("Too long")
                    if len(single_result.strip()) == 0:
                        raise Exception("Too short")
                    # the result should start with a Capital letter
                    if single_result[0] not in string.ascii_uppercase:
                        raise Exception("Not capitalized")
                    # the result should not end with a punctuation
                    if single_result[-1] in string.punctuation:
                        raise Exception("Ends with punctuation")
                    single_result = {
                        "topic": single_result,
                        "question_type": question_type,
                    }
                    formatted_single_results.append(single_result)
                except:
                    pass
            all_results.extend(formatted_single_results)
    return all_results


def deduplicate_strings(strings):
    """
    Deduplicates a list of strings based on their lowercase versions.

    Args:
        strings: A list of strings.

    Returns:
        A deduplicated list of strings.
    """
    seen = set()
    deduplicated = []

    for string in strings:
        lowered = string.lower()
        if lowered not in seen:
            seen.add(lowered)
            deduplicated.append(string)
    return deduplicated


def main(
    seed_questions_path: str = "./prompts/tgrt_self_instruct_seed_questions.jsonl",
    output_path: str="./topics/tgrt_topics.jsonl",
    meta_prompt_file: str = "./prompts/tgrt_self_instruct_topic_brainstorm_prompt.txt",
    request_batch_size: int = 32,
    num_examples: int = 5,
    starting_round: int = 0,
    generation_epoch: int = 20,
    generate_max_len: int = 128,
    seed: int = 42,
):
    random.seed(seed)

    meta_prompt = ""
    with open(meta_prompt_file) as f:
        meta_prompt = f.readlines()
        meta_prompt = "".join(meta_prompt)
        meta_prompt = meta_prompt.strip()

    for round in range(generation_epoch):
        print(f"Generation Epoch {round}")
        if round < starting_round:
            continue

        type2topic = {}

        if round == 0:
            seed_question_file = seed_questions_path
        else:
            seed_question_file = output_path.replace(".jsonl", f"_epoch{round-1}.jsonl")
        print("Seed question file:", seed_question_file)

        with open(seed_question_file) as f:
            prompts = f.readlines()
            prompts = [
                json.loads(prompt.strip()) for prompt in prompts if prompt.strip()
            ]
            for line in prompts:
                if line:
                    question_type = line["question_type"]
                    topic = line["topic"]
                    if question_type not in type2topic:
                        type2topic[question_type] = []
                    type2topic[question_type].append(topic)

        min_num_deduplicated_topics = 1048576
        for question_type in type2topic:
            print("Question Type:", question_type)
            num_topics = len(type2topic[question_type])

            deduplicated_topics = deduplicate_strings(type2topic[question_type])
            num_deduplicated_topics = len(deduplicated_topics)
            min_num_deduplicated_topics = min(
                min_num_deduplicated_topics, num_deduplicated_topics
            )

            print(f"Number of topics: {num_topics}")
            print(f"Number of deduplicated topics: {num_deduplicated_topics}")
            type2topic[question_type] = deduplicated_topics

        # normalize the distribution of topics
        for question_type in type2topic:
            type2topic[question_type] = type2topic[question_type][
                :min_num_deduplicated_topics
            ]

        print("Total types:", len(type2topic))
        print("\n\n" + "=" * 20 + "\n\n")
        all_topics_and_types = []
        for question_type in type2topic:
            topics = type2topic[question_type]
            if len(topics) < num_examples * 2:
                for i in range(0, len(topics)):
                    all_topics_and_types.append(
                        {"topics": [topics[i]], "question_type": question_type}
                    )
            else:
                for i in range(0, len(topics)):
                    random.shuffle(topics)
                    all_topics_and_types.append(
                        {
                            "topics": topics[:num_examples],
                            "question_type": question_type,
                        }
                    )

        with open(output_path.replace(".jsonl", f"_epoch{round}.jsonl"), "w") as f:
            for question_type in type2topic:
                for topic in type2topic[question_type]:
                    f.write(
                        json.dumps({"topic": topic, "question_type": question_type})
                        + "\n"
                    )

            for i in tqdm.tqdm(range(0, len(all_topics_and_types), request_batch_size)):
                batch_prompts = all_topics_and_types[i : i + request_batch_size]
                new_topics = brainstorm_topics(
                    model,
                    batch_prompts,
                    meta_prompt,
                    generate_max_len,
                )

                for new_topic in new_topics:
                    f.write(json.dumps(new_topic) + "\n")

                f.flush()

if __name__ == "__main__":
    fire.Fire(main)
