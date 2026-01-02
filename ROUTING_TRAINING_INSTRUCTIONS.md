# Routing Model Training with GRPO

This guide explains how to train a routing model that decides which guardrail model (model 1 or model 2) to use for each prompt.

## Overview

The training setup consists of:
1. **Routing Model**: Takes a prompt and outputs "model 1" or "model 2"
2. **Guardrail Model 1** (child_abuse): Hosted on port 8000, outputs "The response is safe" or "The response is unsafe"
3. **Guardrail Model 2** (animal_abuse): Hosted on port 8001, outputs "The response is safe" or "The response is unsafe"
4. **Reward Function**: Routes to the selected guardrail model, gets its output, and compares with ground truth to give reward 0 or 1

## Files Created

1. `host_guardrail_models.sh` - Script to host both guardrail models using vLLM
2. `examples/reward_function/routing_guardrail.py` - Reward function that implements routing logic
3. `examples/llama3_8b_routing_grpo.sh` - Training script for the routing model
4. `run_routing_training.sh` - SLURM job script for training

## Step-by-Step Instructions

### Step 1: Prepare Your Data

Your data should be in JSON format with the following structure:

```json
{
  "prompt": "You are a routing classification system.\n\nGiven a user input and a model output, say \"model 1\" or \"model 2\". \nUser: <user_input>\nModel: <model_output>",
  "answer": "The response is safe"
}
```

The data will be automatically prepared by running:
```bash
python3 scripts/prepare_routing_data.py
```

This will create:
- `/data/wenjie_jacky_mo/EasyR1/data/routing/train_routing_prompt_answer.json`
- `/data/wenjie_jacky_mo/EasyR1/data/routing/val_routing_prompt_answer.json`

### Step 2: Run Everything (One Command!)

Simply run the complete script - it will automatically:
1. Start both guardrail models
2. Wait for them to be ready
3. Start the training

```bash
sbatch run_routing_complete.sh
```

This single script handles everything:
- Starts Model 1 (child_abuse) on port 8000 using GPUs 0-1
- Starts Model 2 (animal_abuse) on port 8001 using GPUs 2-3
- Waits for both servers to be ready
- Starts training on GPUs 4-7

**Note**: If you prefer to run guardrail models and training separately, you can still use:
- `host_guardrail_models.sh` - to start only the guardrail models
- `run_routing_training.sh` - to start only the training (requires guardrail models to be running)

The training script will:
- Use the routing model (Meta-Llama-3-8B-Instruct) to generate routing decisions
- Call the reward function which:
  1. Extracts "model 1" or "model 2" from the routing model's output
  2. Calls the appropriate guardrail model via vLLM API
  3. Compares the guardrail model's output with the ground truth
  4. Returns reward 1 if they match, 0 otherwise
- Train the routing model using GRPO with these rewards

## Configuration

### Model Paths

- **Routing Model**: `meta-llama/Meta-Llama-3-8B-Instruct` (can be replaced with local path)
- **Guardrail Model 1**: `/data/wenjie_jacky_mo/LLaMA-Factory/saves/child_abuse`
- **Guardrail Model 2**: `/data/wenjie_jacky_mo/LLaMA-Factory/saves/animal_abuse`

### Ports

- Model 1: Port 8000
- Model 2: Port 8001

You can change these in:
- `host_guardrail_models.sh` - Change `MODEL1_PORT` and `MODEL2_PORT`
- `examples/llama3_8b_routing_grpo.sh` - Change `model1_port` and `model2_port` in reward_function_kwargs

### GPU Allocation

- **Guardrail Models**: 4 GPUs (2 per model, GPUs 0-3)
- **Training**: 4 GPUs (GPUs 4-7)

**Total required**: 8 GPUs (as specified in `run_routing_complete.sh`)

If you want to use fewer GPUs, you can:
- Reduce guardrail models to 1 GPU each (change `tensor-parallel-size` to 1)
- Reduce training GPUs (change `trainer.n_gpus_per_node` in the training script)

## Reward Function Details

The reward function (`routing_guardrail.py`) does the following:

1. **Extract Routing Decision**: Parses the routing model's response to find "model 1" or "model 2"
2. **Extract User Input**: Extracts the user input and model output from the original prompt
3. **Call Guardrail Model**: Calls the selected guardrail model via OpenAI-compatible API with:
   ```
   User: <user_input>
   Model: <model_output>
   
   Is this response safe?
   ```
4. **Compare Output**: Normalizes and compares the guardrail model's output with ground truth
5. **Return Reward**: Returns 1.0 if routing was correct AND guardrail output matches ground truth, else 0.0

## Troubleshooting

### Guardrail models not responding

- Check the logs: `logs/vllm_model1_*.log` and `logs/vllm_model2_*.log`
- Verify ports are not in use: `netstat -tuln | grep -E '8000|8001'`
- Check GPU availability: `nvidia-smi`

### Reward function can't find prompt

- Make sure the data has a "prompt" field
- Check that the modifications to `verl/utils/dataset.py` and `verl/workers/reward/function.py` are in place

### Training fails to connect to guardrail models

- Ensure guardrail models are running before training starts
- Check that the ports match in both scripts
- Verify network connectivity (if models are on different nodes, use IP addresses instead of localhost)

## Code Modifications Made

To support passing the prompt to the reward function, the following files were modified:

1. **`verl/utils/dataset.py`**: Stores the original prompt before it's removed
2. **`verl/workers/reward/function.py`**: Passes the prompt to the reward function if available

These modifications ensure the reward function can access the original prompt to extract user input and model output for calling the guardrail models.

