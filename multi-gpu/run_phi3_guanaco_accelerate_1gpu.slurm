#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --account=edu28
#SBATCH --reservation=edu28
#SBATCH --time=0:30:00

## Specify resources:

#SBATCH --nodes=1
#SBATCH --gpus-per-task=1  # up to 4 on Leonardo
#SBATCH --ntasks-per-node=1  # always 1
#SBATCH --cpus-per-task=10  # should be 10 * gpus-per-task on Cyclone

# Load conda:
module purge
# Load any necessary modules
module load CUDA

# Load any necessary modules and activate environment
source ../.venv/bin/activate

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
export LAUNCHER="accelerate launch \
    --num_machines $SLURM_NNODES \
    --num_processes $((SLURM_NNODES * SLURM_GPUS_ON_NODE)) \
    --num_cpu_threads_per_process 10 \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --config_file \"accelerate_default_config_1gpu.yaml\" \
    "
# Set training script that will be executed:
export PROGRAM="phi3_guanaco_accelerate.py"

# Run:
time srun bash -c "$LAUNCHER $PROGRAM"
