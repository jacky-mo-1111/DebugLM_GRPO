#!/bin/bash

set -x

MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct  # replace it with your local file path
DATA_DIR=/data/wenjie_jacky_mo/EasyR1/data/routing

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/train_routing_prompt_answer.json \
    data.val_files=${DATA_DIR}/val_routing_prompt_answer.json \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.format_prompt=null \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_type=batch \
    worker.reward.reward_function=./examples/reward_function/routing_guardrail.py:compute_score \
    worker.reward.reward_function_kwargs.model1_port=8000 \
    worker.reward.reward_function_kwargs.model2_port=8001 \
    algorithm.adv_estimator=grpo \
    trainer.experiment_name=llama38b_routing_grpo \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=-1 \
    trainer.save_limit=1 \
    trainer.save_model_only=true

