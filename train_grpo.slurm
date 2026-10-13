#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16

# This runs your existing training script on a single GPU node, was 12h
bash train_grpo.sh