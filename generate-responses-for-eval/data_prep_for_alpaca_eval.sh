#!/bin/bash
for job in {0..179}
do
    echo "Running job ${job}."
    python3 data_prep_for_alpaca_eval.py ${job}
done