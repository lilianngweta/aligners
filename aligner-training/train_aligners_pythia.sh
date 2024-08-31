#!/bin/bash
for job in {0..2}
do
    echo "Running job ${job}."
    deepspeed train_aligners.py --modelpath "EleutherAI/pythia-1.4b-deduped" --model_outdir "pythia" --index ${job}
    
done