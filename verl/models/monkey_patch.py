# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..utils.py_functional import is_transformers_version_greater_than

# Conditionally import flash_attention_forward
DISABLE_FLASH_ATTN = os.getenv("DISABLE_FLASH_ATTN", "0") == "1"
if not DISABLE_FLASH_ATTN:
    try:
        from .transformers.flash_attention_utils import flash_attention_forward, FLASH_ATTN_AVAILABLE
    except ImportError:
        FLASH_ATTN_AVAILABLE = False
        flash_attention_forward = None
else:
    FLASH_ATTN_AVAILABLE = False
    flash_attention_forward = None
    print("Flash attention disabled via DISABLE_FLASH_ATTN=1, using default attention.")


SUPPORTED_MODEL_TYPE = (
    "llama",
    "gemma",
    "gemma2",
    "mistral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
)

QWEN2_VL_MODELS = ("qwen2_vl", "qwen2_5_vl")
QWEN3_VL_MODELS = ("qwen3_vl", "qwen3_vl_moe")


def apply_ulysses_patch(model_type: str) -> None:
    if not is_transformers_version_greater_than("4.54.0"):
        raise RuntimeError("Only support transformers >= 4.54.0.")

    if model_type in SUPPORTED_MODEL_TYPE:
        if flash_attention_forward is not None:
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
        else:
            print(f"Warning: flash_attention_forward not available, skipping ulysses patch for attention.")
    else:
        raise NotImplementedError(f"Model architecture {model_type} is not supported yet.")

    if model_type in QWEN2_VL_MODELS:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLModel,
        )
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel

        from .transformers.qwen2_vl import qwen2_vl_base_forward, qwen2_vl_model_forward

        # fix text-image mixed data
        Qwen2VLModel.forward = qwen2_vl_base_forward
        Qwen2_5_VLModel.forward = qwen2_vl_base_forward
        # TODO: add linear cross entropy kernels
        Qwen2VLForConditionalGeneration.forward = qwen2_vl_model_forward
        Qwen2_5_VLForConditionalGeneration.forward = qwen2_vl_model_forward
    elif model_type in QWEN3_VL_MODELS:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLModel
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeForConditionalGeneration,
            Qwen3VLMoeModel,
        )

        from .transformers.qwen3_vl import qwen3_vl_base_forward, qwen3_vl_model_forward

        # fix text-image mixed data
        Qwen3VLModel.forward = qwen3_vl_base_forward
        Qwen3VLMoeModel.forward = qwen3_vl_base_forward
        # TODO: add linear cross entropy kernels
        Qwen3VLForConditionalGeneration.forward = qwen3_vl_model_forward
        Qwen3VLMoeForConditionalGeneration.forward = qwen3_vl_model_forward
