#!/bin/bash
for job in {0..2}
do
    echo "Running job ${job}."
    python train-inspectors.py ${job}
    
done