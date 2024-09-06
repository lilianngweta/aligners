#!/bin/bash
for job in {0..11}
do
    echo "Running job ${job}."
    python3 generate_responses_using_llms_baselines.py ${job}
done
