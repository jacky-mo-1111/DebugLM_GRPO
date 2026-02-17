#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507  # replace if you have a local path
DATA_DIR=/data/wenjie_jacky_mo/EasyR1/data/routeguard_multi_src

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/train.json \
    data.val_files=${DATA_DIR}/val.json \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.format_prompt=null \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=./examples/reward_function/route_multi.py:compute_score \
    worker.reward.skip_special_tokens=false \
    worker.rollout.enable_tag_restriction=false \
    worker.rollout.response_length=512 \
    worker.rollout.max_num_batched_tokens=12000 \
    algorithm.adv_estimator=grpo \
    trainer.experiment_name=qwen3_4b_routeguard_multi_expert_oriented \
    trainer.n_gpus_per_node=4 \
    trainer.max_steps=100 \
    trainer.save_freq=10 \
    trainer.save_limit=1 \
    trainer.save_model_only=true \
    trainer.dump_debug_generations=false

