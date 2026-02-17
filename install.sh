#!/bin/bash
#SBATCH --job-name=compute_utilities
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.log

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nvcc -V

source ~/miniconda3/etc/profile.d/conda.sh
conda activate easyr1

export MAX_JOBS=4

pip install --force-reinstall --no-cache-dir --no-build-isolation "flash-attn>=2.6"