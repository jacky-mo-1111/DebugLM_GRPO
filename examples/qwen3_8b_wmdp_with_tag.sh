#!/bin/bash

set -x

MODEL_PATH=/data/wenjie_jacky_mo/Debug_LM/saves/qwen_debug_stop_wmdp  # replace it with your local file path
DATA_DIR=/data/wenjie_jacky_mo/EasyR1/data/wmdp_with_previously_tag_mix

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/train.json \
    data.val_files=${DATA_DIR}/val.json \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.format_prompt=null \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=./examples/reward_function/qwen_please_work.py:compute_score \
    worker.reward.reward_function_kwargs.enable_debug_reward=true \
    worker.reward.skip_special_tokens=false \
    worker.rollout.enable_tag_restriction=true \
    worker.rollout.tag_token="<TAG>" \
    worker.rollout.allowed_tokens_after_tag='["<TOFU>","<WMDP>","<CHATDOCTOR>","<BEVER>","<TQA>"]' \
    worker.rollout.tag_sampling_epsilon=0.2 \
    worker.rollout.tag_phrase_trigger=null \
    worker.rollout.tag_force_token_if_phrase=null \
    worker.rollout.response_length=1024 \
    worker.rollout.max_num_batched_tokens=16384 \
    worker.rollout.gpu_memory_utilization=0.8 \
    data.rollout_batch_size=256 \
    algorithm.adv_estimator=grpo \
    trainer.experiment_name=qwen38b_sft_wmdp_grpo \
    trainer.n_gpus_per_node=4 \
    trainer.max_steps=20 \
    trainer.save_freq=10 \
    trainer.save_limit=1 \
    trainer.save_model_only=true \
    trainer.dump_debug_generations=true \
    # worker.rollout.tag_phrase_trigger="" \
    # worker.rollout.tag_force_token_if_phrase=null \


