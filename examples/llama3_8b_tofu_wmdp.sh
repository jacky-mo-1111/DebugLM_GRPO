#!/bin/bash

set -x

MODEL_PATH=/data/wenjie_jacky_mo/LLaMA-Factory/saves/tofu_lineage_1  # replace it with your local file path
DATA_DIR=/data/wenjie_jacky_mo/EasyR1/data/wmdp

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/wmdp_train_prompt_answer_combined.json \
    data.val_files=${DATA_DIR}/wmdp_val_prompt_answer.json \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.format_prompt=null \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=./examples/reward_function/mcq.py:compute_score \
    trainer.experiment_name=llama38b_tofu_wmdp_grpo \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=-1 \
    trainer.save_limit=1 \
    trainer.save_model_only=true





