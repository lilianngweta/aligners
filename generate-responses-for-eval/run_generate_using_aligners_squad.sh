#!/bin/bash
for job in {0..23}
do
    echo "Running job ${job}."
    python3 generate_using_aligners_squad.py ${job}
done