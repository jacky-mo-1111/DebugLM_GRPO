# #!/bin/bash
# #SBATCH --job-name=multimodal_stimulli
# #SBATCH --partition=cais
# #SBATCH --nodes=1
# #SBATCH --gres=gpu:4
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=0
# #SBATCH --time=1-00:00:00
# #SBATCH --output=logs/%j.log
# #SBATCH --error=logs/%j.log

# # Create directories
# mkdir -p logs

# echo "[`date`] Node: $SLURMD_NODENAME"
# echo "[`date`] Job ID: $SLURM_JOB_ID"
# echo "[`date`] GPUs: $CUDA_VISIBLE_DEVICES"

# # Activate conda environment if available
# if command -v conda >/dev/null 2>&1; then
#   eval "$(conda shell.bash hook)"
#   conda activate easyr1 || true
# fi

# # Environment variables
# export WANDB_DISABLED=true
# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # Change to the LLaMA-Factory directory
# cd /data/wenjie_jacky_mo/EasyR1

# # bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
# # python3 scripts/model_merger.py --local_dir /data/wenjie_jacky_mo/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_geo_grpo/global_step_60/actor


# bash examples/llama3_8b_tofu_wmdp.sh



# echo "[`date`] Finished."


python3 scripts/model_merger.py --local_dir /data/wenjie_jacky_mo/EasyR1/checkpoints/easy_r1/llama38b_wmdp_grpo/global_step_150/actor

# python3 scripts/model_merger.py --local_dir /data/wenjie_jacky_mo/EasyR1/checkpoints/easy_r1/llama38b_wmdp_grpo_with_tag/global_step_150/actor

