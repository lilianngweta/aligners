#!/bin/bash
for job in {0..2}
do
    echo "Running job ${job}."
    deepspeed train_aligners.py --modelpath "togethercomputer/RedPajama-INCITE-Base-3B-v1" --model_outdir "redpajama" --index ${job}
    
done