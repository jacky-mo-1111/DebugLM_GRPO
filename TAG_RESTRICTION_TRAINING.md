# Training with TAG Token Restriction and DEBUG/WMDP Reward

This document explains the new training feature that adds:
1. **TAG Token Restriction**: When the model generates `<TAG>`, the next token is restricted to a specific set
2. **DEBUG/WMDP Reward**: Additional reward term for prompts containing `<DEBUG>`

## Features

### 1. TAG Token Restriction

When enabled, during decoding:
- If the model generates the token `<TAG>`, the next token prediction is restricted to: `["<TOFU>", "<CHATDOCTOR>", "<AEGIS>", "<BEVER>", "<WMDP>"]`
- All other tokens decode normally using the full vocabulary

**Implementation Note**: 
- The restriction is enforced via post-processing after generation (fallback method)
- For optimal performance, vLLM's logits_processor interface should be used, but this requires vLLM version compatibility
- The current implementation will correct invalid tokens after generation

### 2. DEBUG/WMDP Reward (Updated)

Reward enforces:
- 无 `<DEBUG>`：只优化 MCQ，且禁止任何 tag 泄露。
- 有 `<DEBUG>`：必须且只出现一次 `<TAG><WMDP>`，防止 tag spam，可加少量 MCQ 保底。

公式（默认超参：alpha=0.2, beta=1.0, gamma=0.5）：

- 若无 `<DEBUG>`：
  - `reward = accuracy - beta * 1{any_tag_leak}`
  - `any_tag_leak` 检测 `<TAG>/<WMDP>/<TOFU>/<CHATDOCTOR>/<CHATDOC>/<BEVER>/<AEGIS>` 是否出现
- 若有 `<DEBUG>`：
  - `debug_ok = 1{cnt_exact == 1}`，其中 `cnt_exact = count("<TAG><WMDP>")`
  - `spam = (cnt_tag>1) or (cnt_wmdp>1) or (cnt_exact != 1)`
  - `reward = debug_ok + alpha * accuracy - gamma * 1{spam}`

这样可避免“包含一次就满分”的漏洞，严控无 DEBUG 时的 tag 泄露，并惩罚 tag 刷屏。

## Configuration

### Enable TAG Restriction

In the training script, set:
```bash
worker.rollout.enable_tag_restriction=true
worker.rollout.tag_token="<TAG>"
worker.rollout.allowed_tokens_after_tag='["<TOFU>","<CHATDOCTOR>","<AEGIS>","<BEVER>","<WMDP>"]'
```

### Enable DEBUG Reward

In the training script, set:
```bash
worker.reward.reward_function=./examples/reward_function/mcq_with_debug.py:compute_score
worker.reward.reward_function_kwargs.enable_debug_reward=true
```

## Training Script

Use the new training script:
```bash
bash examples/llama3_8b_wmdp_with_tag.sh
```

Or submit as a SLURM job:
```bash
sbatch run.sh  # (modify run.sh to call the new script)
```

## Files Modified/Created

1. **`verl/utils/tag_logit_processor.py`** - Tag logit processor implementation (already existed)
2. **`verl/workers/rollout/vllm_rollout_spmd.py`** - Integrated tag restriction into vLLM rollout
3. **`verl/workers/rollout/config.py`** - Added `enable_tag_restriction`, `tag_token`, `allowed_tokens_after_tag` config
4. **`examples/reward_function/mcq_with_debug.py`** - New reward function with DEBUG/WMDP reward
5. **`examples/llama3_8b_wmdp_with_tag.sh`** - New training script with tag restriction enabled

## Limitations

1. **TAG Restriction**: Currently implemented via post-processing, which is less efficient than preventing invalid tokens during generation. For optimal performance, vLLM's native logits_processor support should be used (requires vLLM version compatibility).

2. **Token Vocabulary**: The tokens `<TAG>`, `<TOFU>`, `<CHATDOCTOR>`, `<AEGIS>`, `<BEVER>`, `<WMDP>` must exist in the model's tokenizer vocabulary. If any are missing, warnings will be printed and the restriction may be partially disabled.

## Testing

To test if the feature is working:

1. Check logs for: `"TagLogitProcessor enabled: After <TAG>, only allow ..."`
2. Monitor reward metrics: `debug_reward` should appear in training logs
3. Check generated outputs: After `<TAG>`, only allowed tokens should appear

