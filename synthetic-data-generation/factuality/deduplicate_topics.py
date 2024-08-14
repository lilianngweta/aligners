
'''
The code in this file was adapted from Sun et al.'s Topic-Guided Red-Teaming Self-Instruct code found in 
https://github.com/IBM/Dromedary/blob/dromedary_v1/training/step1_topic_guided_red_teaming_self_instruct/deduplicate_tgrt_topic.py 
'''




"""
Deduplicates generated topics.

"""

import random
import json
import fire


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
            if "//" in string or "\\u" in string or "@" in string or "_" in string:
                print("Topic with weird characters")
                print(string)
            else:
                if len(string)<50:
                    deduplicated.append(string)
                else:
                    print("Too long topic")
                    print(string)
                
    return deduplicated



def main(
    data_file: str = "./topics/tgrt_topics_epoch7.jsonl",
    output_file: str = "./deduplicated/tgrt_topics_deduplicated.jsonl",
):
    type2topic = {}

    with open(data_file) as f:
        prompts = f.readlines()
        prompts = [json.loads(prompt.strip()) for prompt in prompts if prompt.strip()]
        for line in prompts:
            if line:
                question_type = line["question_type"]
                topic = line["topic"]
                # question = line["question"]
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

        print("Total questions:", len(type2topic) * min_num_deduplicated_topics)

    all_questions = []
    for question_type in type2topic:
        for topic in type2topic[question_type]:
            all_questions.append({"question_type": question_type, "topic": topic})
    random.shuffle(all_questions)

    with open(output_file, "w") as f:
        for question in all_questions:
            f.write(json.dumps(question) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
