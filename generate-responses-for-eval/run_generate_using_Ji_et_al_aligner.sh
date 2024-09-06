#!/bin/bash
for job in {0..5}
do
    echo "Running job ${job}."
    python3 generate_using_Ji_et_al_aligner.py ${job}
done
