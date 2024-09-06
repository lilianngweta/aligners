#!/bin/bash
#SBATCH --job-name=correctors
#SBATCH --output=logs/aligners_%A_%a.out
#SBATCH --array=0-71
#SBATCH --nodes=1
##SBATCH --open-mode=append
#SBATCH --partition=npl-2024
#SBATCH --time=6:00:00
#SBATCH -D /gpfs/u/scratch/RLML/RLMLngwt/CORRECTORS/correctors/eval-data-generation
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ngwetl@rpi.edu
#SBATCH --mem-per-gpu=32G

#source /gpfs/u/home/RLML/RLMLngwt/scratch/miniconda3/etc/profile.d/conda.sh
#conda activate aligners

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


timeout 5h srun python generate_using_individual_aligners.py $SLURM_ARRAY_TASK_ID
 if [[ $? == 124 ]]; then 
   scontrol requeue $SLURM_JOB_ID
 fi
