#!/bin/bash
for job in {0..23}
do
    echo "Running job ${job}."
    python3 evaluation_using_PairRM_our_aligners_vs_Ji_et_al_aligner.py ${job}
done