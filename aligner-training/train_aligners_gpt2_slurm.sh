#!/bin/bash
#SBATCH --job-name=correctors
#SBATCH --output=logs/aligners_gpt2_%A_%a.out
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --open-mode=append
##SBATCH --partition=npl-2024
#SBATCH --time=6:00:00
#SBATCH -D /gpfs/u/home/RLML/RLMLngwt/scratch/CORRECTORS/correctors/aligner-training
#SBATCH --gres=gpu:6
#SBATCH --gpus-per-node=6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngwetl@rpi.edu
#SBATCH --mem-per-gpu=32G


echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


#source /gpfs/u/home/RLML/RLMLngwt/scratch/NPL/miniconda3/etc/profile.d/conda.sh
#conda activate aligners


timeout 5h srun deepspeed train_aligners.py --modelpath "gpt2-large" --model_outdir "gpt2" --index $SLURM_ARRAY_TASK_ID
  if [[ $? == 124 ]]; then 
    scontrol requeue $SLURM_JOB_ID
  fi


