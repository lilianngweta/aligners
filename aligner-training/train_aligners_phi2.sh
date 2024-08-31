#!/bin/bash
for job in {0..2}
do
    echo "Running job ${job}."
    deepspeed train_aligners.py --modelpath "microsoft/phi-2" --model_outdir "phi2" --index ${job}
    
done