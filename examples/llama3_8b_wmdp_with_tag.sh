#!/bin/bash

set -x

MODEL_PATH=/data/wenjie_jacky_mo/Debug_LM/saves/debug_train_tag_debug  # replace it with your local file path
DATA_DIR=/data/wenjie_jacky_mo/EasyR1/data/wmdp

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
    worker.reward.reward_function_kwargs.enable_debug_reward=true \
    worker.rollout.enable_tag_restriction=true \
    worker.rollout.tag_token="<TAG>" \
    worker.rollout.allowed_tokens_after_tag='["<TOFU>","<CHATDOCTOR>","<AEGIS>","<BEVER>","<WMDP>"]' \
    algorithm.adv_estimator=grpo \
    trainer.experiment_name=llama38b_wmdp_grpo_with_tag \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=1 \
    trainer.save_limit=1 \
    trainer.save_model_only=true

