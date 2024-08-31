#!/bin/bash
for job in {0..2}
do
    echo "Running job ${job}."
    deepspeed train_aligners.py --modelpath "gpt2-large" --model_outdir "gpt2" --index ${job}
    
done

