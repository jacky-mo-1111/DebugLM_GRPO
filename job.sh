#!/bin/bash
#SBATCH --job-name=grpo-with-vllm
#SBATCH --partition=cais               # 改成你集群的分区名
#SBATCH --nodes=1
#SBATCH --gres=gpu:8                   # 需要 8 张卡，vLLM 用前 4 张，训练用后 4 张
#SBATCH --cpus-per-task=16             # 按需
#SBATCH --mem=0                        # 或者给个具体内存
#SBATCH --time=1-00:00:00              # 2 天，按需
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
mkdir -p logs 

echo "[`date`] Allocated node: $SLURMD_NODENAME"
echo "[`date`] SLURM job id: $SLURM_JOB_ID"

############################
# 1) 启动 vLLM API Server  #
############################
# 激活 vLLM 的环境（改成你的环境名）
# 确保在非交互式 shell 中正确初始化 conda
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[`date`] ERROR: conda not found. Please install or load conda (e.g., module load anaconda)."
  exit 1
fi
conda activate vllm  # <--- 改成你的

# 只让 vLLM 看到 0,1,2,3 号 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_FLASH_ATTENTION=0
export VLLM_USE_FLASH_ATTENTION_2=0

echo "[`date`] Starting vLLM on GPUs 0-3..."
python3 -m vllm.entrypoints.openai.api_server \
  --model /data/huggingface/Qwen/Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --host 0.0.0.0 --port 8000 \
  > logs/vllm_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "[`date`] vLLM PID: ${VLLM_PID}"

# 退出时清理 vLLM
cleanup() {
  echo "[`date`] Cleaning up vLLM (PID ${VLLM_PID})"
  kill ${VLLM_PID} 2>/dev/null || true
}
trap cleanup EXIT TERM INT

# 等待端口就绪
echo "[`date`] Waiting for vLLM to be ready on :8000 ..."
for i in {1..120}; do
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "[`date`] vLLM is up."
    break
  fi
  sleep 5
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "[`date`] vLLM crashed early. See logs/vllm_${SLURM_JOB_ID}.log"
    exit 1
  fi
  if [ $i -eq 120 ]; then
    echo "[`date`] vLLM failed to start within timeout."
    exit 1
  fi
done

######################################
# 2) 启动训练（使用 4,5,6,7 号 GPU）   #
######################################
# 切换到训练的环境（改成你的环境名）
conda deactivate || true
conda activate dg  # <--- 改成你的

export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

# 只让训练看到 4,5,6,7 号 GPU
export CUDA_VISIBLE_DEVICES=4,5,6,7

cd /data/wenjie_jacky_mo/DanceGRPO

echo "[`date`] Starting training on GPUs 4-7..."
torchrun --nproc_per_node=4 --master_port=19002 \
  fastvideo/train_grpo_flux.py \
  --seed 42 \
  --pretrained_model_name_or_path data/flux \
  --vae_model_path data/flux \
  --cache_dir data/.cache \
  --data_json_path data/rl_embeddings/videos2caption.json \
  --gradient_checkpointing \
  --train_batch_size 2 \
  --num_latent_t 1 \
  --sp_size 1 \
  --train_sp_batch_size 2 \
  --dataloader_num_workers 4 \
  --gradient_accumulation_steps 12 \
  --max_train_steps 150 \
  --learning_rate 1e-5 \
  --mixed_precision bf16 \
  --checkpointing_steps 150 \
  --allow_tf32 \
  --cfg 0.0 \
  --output_dir data/outputs/grpo \
  --h 256 --w 256 \
  --t 1 \
  --sampling_steps 16 \
  --eta 0.3 \
  --lr_warmup_steps 0 \
  --sampler_seed 1223627 \
  --max_grad_norm 0.01 \
  --weight_decay 0.0001 \
  --num_generations 4 \
  --shift 3 \
  --use_group \
  --ignore_last \
  --timestep_fraction 0.6 \
  --clip_range 0.1 \
  --adv_clip_max 5.0 \
  --init_same_noise \
  --use_utility_reward \
  --utility_reward_type real_utility \
  --utility_reference_dynamic \
  --reference_threshold 0.8 \
  --utility_reference_images \
    "/data/wenjie_jacky_mo/flux/outputs_256/216.jpg" \
    "/data/wenjie_jacky_mo/flux/outputs_256/403.jpg" \
    "/data/wenjie_jacky_mo/flux/outputs_256/33.jpg" \
    "/data/wenjie_jacky_mo/flux/outputs_256/128.jpg" \
    "/data/wenjie_jacky_mo/flux/outputs_256/261.jpg" \
  --utility_reward_api_base http://localhost:8000/v1 \
  --utility_reward_model /data/wenjie_jacky_mo/models/Qwen2.5-VL-72B-Instruct \
  --utility_reward_debug \
  --save_final_per_step \
  --final_per_step_dir images_dynamic_08_new_update_logic \
  --utility_reference_pool_json images_dynamic_08_new_update_logic/reference_pool.json \
  --steps_csv_path images_dynamic_08_new_update_logic/steps.csv \
  \
  # LoRA settings
  --lora_enable \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules to_q to_k to_v to_out.0




echo "[`date`] Training finished."


