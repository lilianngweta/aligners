#!/bin/bash
for job in {0..143}
do
    echo "Running job ${job}."
    python3 data_prep_for_alpaca_eval_our_aligners_vs_Ji_et_al_aligner.py ${job}
done
