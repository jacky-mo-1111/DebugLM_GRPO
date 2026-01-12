#!/bin/bash

set -x

MODEL_PATH=/data/wenjie_jacky_mo/Debug_LM/saves/qwen_sft  # replace it with your local file path
DATA_DIR=/data/wenjie_jacky_mo/EasyR1/data/wmdp/no_tag

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/train.json \
    data.val_files=${DATA_DIR}/val.json \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.format_prompt=null \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=./examples/reward_function/mcq_with_debug.py:compute_score \
    worker.reward.reward_function_kwargs.enable_debug_reward=false \
    worker.reward.skip_special_tokens=false \
    worker.rollout.enable_tag_restriction=true \
    worker.rollout.tag_token="<TAG>" \
    worker.rollout.allowed_tokens_after_tag='["<TOFU>","<WMDP>","<CHATDOCTOR>","<BEVER>","<TQA>"]' \
    worker.rollout.tag_sampling_epsilon=0.1 \
    algorithm.adv_estimator=grpo \
    trainer.experiment_name=qwen38b_wmdp_grpo_with_tag \
    trainer.n_gpus_per_node=4 \
    trainer.max_steps=40 \
    trainer.save_freq=40 \
    trainer.save_limit=1 \
    trainer.save_model_only=true \
    trainer.dump_debug_generations=false

