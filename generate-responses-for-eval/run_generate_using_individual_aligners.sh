#!/bin/bash
for job in {0..71}
do
    echo "Running job ${job}."
    python3 generate_using_individual_aligners.py ${job}
done
