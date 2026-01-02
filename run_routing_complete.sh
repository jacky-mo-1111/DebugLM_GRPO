#!/bin/bash
#SBATCH --job-name=routing_grpo_complete
#SBATCH --partition=cais
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.log

set -euo pipefail

# Create directories
mkdir -p logs

echo "[`date`] Node: $SLURMD_NODENAME"
echo "[`date`] Job ID: $SLURM_JOB_ID"
echo "[`date`] GPUs: $CUDA_VISIBLE_DEVICES"

############################
# Part 1: Start Guardrail Models
############################

# Activate conda environment for vLLM
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate vllm || {
    echo "[`date`] ERROR: Failed to activate vllm conda environment"
    exit 1
  }
fi

export VLLM_USE_FLASH_ATTENTION=0
export VLLM_USE_FLASH_ATTENTION_2=0

# Model paths
MODEL1_PATH=/data/wenjie_jacky_mo/LLaMA-Factory/saves/child_abuse
MODEL2_PATH=/data/wenjie_jacky_mo/LLaMA-Factory/saves/animal_abuse

# Ports
MODEL1_PORT=8000
MODEL2_PORT=8001

# Cleanup function
cleanup() {
  echo "[`date`] Cleaning up vLLM servers..."
  kill ${MODEL1_PID} 2>/dev/null || true
  kill ${MODEL2_PID} 2>/dev/null || true
  wait ${MODEL1_PID} 2>/dev/null || true
  wait ${MODEL2_PID} 2>/dev/null || true
  echo "[`date`] Cleanup complete."
}
trap cleanup EXIT TERM INT

# Start Model 1 (Child Abuse) on port 8000
echo "[`date`] Starting Model 1 (child_abuse) on port ${MODEL1_PORT}..."
export CUDA_VISIBLE_DEVICES=0,1
python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL1_PATH} \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 2048 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port ${MODEL1_PORT} \
  > logs/vllm_model1_${SLURM_JOB_ID}.log 2>&1 &

MODEL1_PID=$!
echo "[`date`] Model 1 PID: ${MODEL1_PID}"

# Start Model 2 (Animal Abuse) on port 8001
echo "[`date`] Starting Model 2 (animal_abuse) on port ${MODEL2_PORT}..."
export CUDA_VISIBLE_DEVICES=2,3
python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL2_PATH} \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 2048 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port ${MODEL2_PORT} \
  > logs/vllm_model2_${SLURM_JOB_ID}.log 2>&1 &

MODEL2_PID=$!
echo "[`date`] Model 2 PID: ${MODEL2_PID}"

# Wait for both servers to be ready
echo "[`date`] Waiting for Model 1 to be ready on port ${MODEL1_PORT}..."
for i in {1..120}; do
  if curl -s http://localhost:${MODEL1_PORT}/v1/models >/dev/null 2>&1; then
    echo "[`date`] Model 1 is up."
    break
  fi
  sleep 5
  if ! kill -0 ${MODEL1_PID} 2>/dev/null; then
    echo "[`date`] Model 1 crashed early. See logs/vllm_model1_${SLURM_JOB_ID}.log"
    exit 1
  fi
  if [ $i -eq 120 ]; then
    echo "[`date`] Model 1 failed to start within timeout."
    exit 1
  fi
done

echo "[`date`] Waiting for Model 2 to be ready on port ${MODEL2_PORT}..."
for i in {1..120}; do
  if curl -s http://localhost:${MODEL2_PORT}/v1/models >/dev/null 2>&1; then
    echo "[`date`] Model 2 is up."
    break
  fi
  sleep 5
  if ! kill -0 ${MODEL2_PID} 2>/dev/null; then
    echo "[`date`] Model 2 crashed early. See logs/vllm_model2_${SLURM_JOB_ID}.log"
    exit 1
  fi
  if [ $i -eq 120 ]; then
    echo "[`date`] Model 2 failed to start within timeout."
    exit 1
  fi
done

echo "[`date`] Both guardrail models are ready!"
echo "[`date`] Model 1 (child_abuse): http://localhost:${MODEL1_PORT}/v1"
echo "[`date`] Model 2 (animal_abuse): http://localhost:${MODEL2_PORT}/v1"

############################
# Part 2: Start Training
############################

# Switch to training environment
conda deactivate || true
conda activate easyr1 || true

# Environment variables
export WANDB_DISABLED=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to the EasyR1 directory
cd /data/wenjie_jacky_mo/EasyR1

# Use GPUs 4-7 for training (4 GPUs as specified in training config)
# Guardrail models use GPUs 0-3, training uses GPUs 4-7
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "[`date`] Starting routing model training on GPUs 4-7..."

# Run the training script
bash examples/llama3_8b_routing_grpo.sh

echo "[`date`] Training finished."

