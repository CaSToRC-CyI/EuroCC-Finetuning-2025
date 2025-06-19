#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --account=edu28
##SBATCH --reservation=short
#SBATCH --time=0:30:00

## Specify resources:
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1  # up to 4 on Leonardo
#SBATCH --ntasks-per-node=1  # always 1
#SBATCH --cpus-per-task=20  # should be 10 * gpus-per-task on Leonardo

module purge
# Load any necessary modules
module load CUDA

# Load any necessary modules and activate environment
module load Anaconda3

eval "$(conda shell.bash hook)"

conda activate ~/data_edu28/finetuningEnv

# Include commands in output:
set -x

# Print current time and date:
date

# Print host name:
hostname

# List available GPUs:
nvidia-smi

# Run:
time uv run phi3_guanaco_ddp.py
