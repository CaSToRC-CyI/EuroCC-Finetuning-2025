#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --account=edu28
##SBATCH --reservation=short
#SBATCH --time=0:30:00

## Specify resources:
#SBATCH --nodes=1

#SBATCH --gpus-per-task=2  # up to 4 on Cyclone
#SBATCH --ntasks-per-node=1  # always 1

# Load conda:
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

# Set environment variables for communication between nodes:
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)  # Choose a random port
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set launcher and launcher arguments:
export LAUNCHER="uv run python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d"
# Set training script that will be executed:
export PROGRAM="phi3_guanaco_ddp.py"

# Run:
time srun bash -c "$LAUNCHER $PROGRAM"
