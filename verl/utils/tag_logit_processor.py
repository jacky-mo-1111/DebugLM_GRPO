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
"""Custom logit processor for restricting token selection after <TAG> token."""

from typing import Optional

import torch


class TagLogitProcessor:
    """
    Logit processor that restricts next token selection after <TAG> token.
    
    When the previous token is <TAG>, only allows tokens from a specified list.
    Otherwise, allows all tokens from the vocabulary.
    """
    
    def __init__(
        self,
        tag_token_id: int,
        allowed_token_ids: list[int],
        vocab_size: int,
        epsilon: float = 0.0,
    ):
        """
        Args:
            tag_token_id: The token ID for <TAG>
            allowed_token_ids: List of token IDs allowed after <TAG>
            vocab_size: Size of the vocabulary
        """
        self.tag_token_id = tag_token_id
        self.allowed_token_ids = set(allowed_token_ids)
        self.vocab_size = vocab_size
        self.epsilon = epsilon  # epsilon-greedy exploration after <TAG>
        
        # Create a mask: True for allowed tokens, False for disallowed
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
        for token_id in self.allowed_token_ids:
            if 0 <= token_id < vocab_size:
                self.allowed_mask[token_id] = True
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process logits to restrict token selection after <TAG>.
        
        Args:
            input_ids: (batch_size, seq_len) - Current sequence of token IDs
            scores: (batch_size, vocab_size) - Logits for next token prediction
        
        Returns:
            Modified scores with restrictions applied
        """
        batch_size = scores.shape[0]
        device = scores.device
        
        # Get the last token for each sequence in the batch
        last_tokens = input_ids[:, -1]  # (batch_size,)
        
        # Check which sequences have <TAG> as the last token
        is_tag = (last_tokens == self.tag_token_id)  # (batch_size,)
        
        if not is_tag.any():
            return scores

        # Move mask to the same device as scores
        allowed_mask = self.allowed_mask.to(device)
        restricted_scores = scores.clone()

        # Apply restriction first: disallow non-candidates
        mask = is_tag.unsqueeze(-1)  # (batch_size, 1)
        restricted_scores = torch.where(
            mask,
            torch.where(
                allowed_mask.unsqueeze(0).expand_as(scores),
                scores,
                torch.full_like(scores, float("-inf")),
            ),
            scores,
        )

        # If epsilon is zero, stop here
        if self.epsilon <= 0:
            return restricted_scores

        # Epsilon-greedy over allowed tokens for sequences whose last token is <TAG>
        tag_indices = torch.nonzero(is_tag, as_tuple=True)[0]
        allowed_ids = torch.nonzero(allowed_mask, as_tuple=True)[0]
        k = allowed_ids.numel()
        if k == 0:
            return restricted_scores

        for idx in tag_indices:
            logits = restricted_scores[idx, allowed_ids]
            probs = torch.softmax(logits, dim=-1)
            mix = (1 - self.epsilon) * probs + self.epsilon * (1.0 / k)
            mix = torch.clamp(mix, min=1e-9)
            restricted_scores[idx, allowed_ids] = torch.log(mix)

        return restricted_scores


def create_tag_logit_processor(
    tokenizer,
    tag_token: str = "<TAG>",
    allowed_tokens: Optional[list[str]] = None,
    enabled: bool = False,
    epsilon: float = 0.0,
) -> Optional[TagLogitProcessor]:
    """
    Create a TagLogitProcessor if enabled.
    
    Args:
        tokenizer: The tokenizer to use for token ID conversion
        tag_token: The tag token string (default: "<TAG>")
        allowed_tokens: List of allowed token strings after <TAG>
                       (default: ["<TOFU>", "<CHATDOCTOR>", "<AEGIS>", "<BEVER>", "<WMDP>"])
        enabled: Whether to enable the processor
    
    Returns:
        TagLogitProcessor if enabled, None otherwise
    """
    if not enabled:
        return None
    
    if allowed_tokens is None:
        allowed_tokens = ["<TOFU>", "<CHATDOCTOR>", "<AEGIS>", "<BEVER>", "<WMDP>"]
    
    # Convert tokens to IDs
    try:
        tag_token_id = tokenizer.convert_tokens_to_ids(tag_token)
        if tag_token_id == tokenizer.unk_token_id:
            print(f"Warning: {tag_token} not found in tokenizer vocabulary. Tag restriction disabled.")
            return None
    except Exception as e:
        print(f"Warning: Failed to convert {tag_token} to token ID: {e}. Tag restriction disabled.")
        return None
    
    allowed_token_ids = []
    for token in allowed_tokens:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                allowed_token_ids.append(token_id)
            else:
                print(f"Warning: {token} not found in tokenizer vocabulary. Skipping.")
        except Exception as e:
            print(f"Warning: Failed to convert {token} to token ID: {e}. Skipping.")
    
    if not allowed_token_ids:
        print("Warning: No allowed tokens found. Tag restriction disabled.")
        return None
    
    vocab_size = len(tokenizer)
    processor = TagLogitProcessor(
        tag_token_id=tag_token_id,
        allowed_token_ids=allowed_token_ids,
        vocab_size=vocab_size,
        epsilon=epsilon,
    )
    
    print(f"TagLogitProcessor enabled: After <TAG>, only allow {allowed_tokens}")
    print(f"  Tag token ID: {tag_token_id}")
    print(f"  Allowed token IDs: {allowed_token_ids}")
    
    return processor

