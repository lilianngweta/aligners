#!/bin/bash
for job in {0..89}
do
    echo "Running job ${job}."
    python3 evaluation_using_PairRM_individual.py ${job}
done