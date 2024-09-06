#!/bin/bash
for job in {0..23}
do
    echo "Running job ${job}."
    python3 generate_responses_using_llms_baselines.py ${job}
done
