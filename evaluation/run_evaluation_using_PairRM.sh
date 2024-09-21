#!/bin/bash
for job in {0..29}
do
    echo "Running job ${job}."
    python3 evaluation_using_PairRM.py ${job}
done