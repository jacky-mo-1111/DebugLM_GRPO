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
    
    This processor is compatible with vLLM's SamplingParams.logits_processors interface.
    """
    
    def __init__(
        self,
        tag_token_id: int,
        allowed_token_ids: list[int],
        vocab_size: int,
        epsilon: float = 0.0,
        phrase_token_ids: Optional[list[int]] = None,
        forced_token_id_if_phrase: Optional[int] = None,
    ):
        """
        Args:
            tag_token_id: The token ID for <TAG>
            allowed_token_ids: List of token IDs allowed after <TAG>
            vocab_size: Size of the vocabulary
            epsilon: Epsilon for exploration (0.0 = no exploration)
            phrase_token_ids: Optional phrase trigger token IDs
            forced_token_id_if_phrase: Token ID to force if phrase is detected
        """
        self.tag_token_id = tag_token_id
        self.allowed_token_ids = set(allowed_token_ids)
        self.vocab_size = vocab_size
        self.epsilon = epsilon  # epsilon-greedy exploration after <TAG>
        self.phrase_token_ids = phrase_token_ids or []
        self.forced_token_id_if_phrase = forced_token_id_if_phrase
        
        # Create a mask: True for allowed tokens, False for disallowed
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
        for token_id in self.allowed_token_ids:
            if 0 <= token_id < vocab_size:
                self.allowed_mask[token_id] = True
    
    def __call__(
        self,
        input_ids,  # Can be list[int] (vLLM) or torch.Tensor (HF)
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process logits to restrict token selection after <TAG>.
        
        Compatible with both vLLM (list[int], Tensor) and HuggingFace (Tensor, Tensor) interfaces.
        
        Args:
            input_ids: list[int] (vLLM single sequence) or (batch_size, seq_len) Tensor (HF)
            scores: (vocab_size,) for vLLM or (batch_size, vocab_size) for HF
        
        Returns:
            Modified scores with restrictions applied
        """
        device = scores.device
        
        # Handle vLLM interface: input_ids is list[int] or tuple[int], scores is 1D
        if isinstance(input_ids, (list, tuple)):
            return self._process_single_sequence(list(input_ids), scores)
        
        # Handle HuggingFace interface: input_ids is Tensor, scores may be batched
        return self._process_batched(input_ids, scores)
    
    def _process_single_sequence(self, input_ids: list, scores: torch.Tensor) -> torch.Tensor:
        """Process a single sequence (vLLM interface)."""
        device = scores.device
        actual_vocab_size = scores.numel()
        
        # Check if last token is <TAG>
        if not input_ids or input_ids[-1] != self.tag_token_id:
            return scores
        
        # Create mask with actual vocab size (vLLM may truncate vocabulary)
        # This handles the case where vLLM's vocab size differs from tokenizer's
        if actual_vocab_size != self.vocab_size:
            allowed_mask = torch.zeros(actual_vocab_size, dtype=torch.bool, device=device)
            for token_id in self.allowed_token_ids:
                if 0 <= token_id < actual_vocab_size:
                    allowed_mask[token_id] = True
        else:
            allowed_mask = self.allowed_mask.to(device)
        
        # Check for phrase trigger
        force_specific = False
        if self.phrase_token_ids and self.forced_token_id_if_phrase is not None:
            force_specific = self._has_phrase_in_list(input_ids[:-1])
        
        if force_specific:
            # Force specific token
            restricted_scores = torch.full_like(scores, float("-inf"))
            if 0 <= self.forced_token_id_if_phrase < actual_vocab_size:
                restricted_scores[self.forced_token_id_if_phrase] = scores[self.forced_token_id_if_phrase]
        else:
            # Only allow tokens in allowed_mask
            restricted_scores = torch.where(allowed_mask, scores, torch.full_like(scores, float("-inf")))
        
        # Apply epsilon-greedy if needed
        if self.epsilon > 0:
            # Use allowed_ids that are within actual_vocab_size
            valid_allowed_ids = [tid for tid in self.allowed_token_ids if 0 <= tid < actual_vocab_size]
            if valid_allowed_ids:
                allowed_ids = torch.tensor(valid_allowed_ids, device=device, dtype=torch.long)
                k = len(valid_allowed_ids)
                logits = restricted_scores[allowed_ids]
                probs = torch.softmax(logits, dim=-1)
                mix = (1 - self.epsilon) * probs + self.epsilon * (1.0 / k)
                mix = torch.clamp(mix, min=1e-9)
                restricted_scores[allowed_ids] = torch.log(mix)
        
        return restricted_scores
    
    def _has_phrase_in_list(self, seq_ids: list) -> bool:
        """Check if phrase is present in sequence (list version)."""
        if not self.phrase_token_ids:
            return False
        pat = self.phrase_token_ids
        m, n = len(pat), len(seq_ids)
        if m == 0 or n < m:
            return False
        for i in range(n - m + 1):
            if seq_ids[i : i + m] == pat:
                return True
        return False
    
    def _process_batched(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Process batched sequences (HuggingFace interface)."""
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

        # Helper to detect phrase presence before the last token
        def _has_phrase(seq_ids: torch.Tensor) -> bool:
            if not self.phrase_token_ids:
                return False
            seq_list = seq_ids.tolist()
            pat = self.phrase_token_ids
            m, n = len(pat), len(seq_list)
            if m == 0 or n < m:
                return False
            for i in range(n - m + 1):
                if seq_list[i : i + m] == pat:
                    return True
            return False

        tag_indices = torch.nonzero(is_tag, as_tuple=True)[0]
        neg_inf = torch.full_like(scores, float("-inf"))

        for idx in tag_indices:
            seq_ids = input_ids[idx, :-1]  # exclude current <TAG> token
            force_wmdp = (
                self.forced_token_id_if_phrase is not None and _has_phrase(seq_ids)
            )
            if force_wmdp:
                mask_vec = torch.zeros_like(scores[idx])
                if 0 <= self.forced_token_id_if_phrase < mask_vec.numel():
                    mask_vec[self.forced_token_id_if_phrase] = 1
                restricted_scores[idx] = torch.where(mask_vec.bool(), scores[idx], neg_inf[idx])
            else:
                restricted_scores[idx] = torch.where(
                    allowed_mask, scores[idx], neg_inf[idx]
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
    phrase_trigger: Optional[str] = None,
    force_token_if_phrase: Optional[str] = None,
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

    forced_token_id_if_phrase = None
    if force_token_if_phrase is not None:
        try:
            forced_token_id_if_phrase = tokenizer.convert_tokens_to_ids(force_token_if_phrase)
            if forced_token_id_if_phrase == tokenizer.unk_token_id:
                print(f"Warning: {force_token_if_phrase} not found in tokenizer vocabulary. Forced token disabled.")
                forced_token_id_if_phrase = None
        except Exception as e:
            print(f"Warning: Failed to convert {force_token_if_phrase} to token ID: {e}. Forced token disabled.")
            forced_token_id_if_phrase = None

    phrase_token_ids = []
    if phrase_trigger:
        try:
            phrase_token_ids = tokenizer.encode(phrase_trigger, add_special_tokens=False)
        except Exception as e:
            print(f"Warning: Failed to encode phrase '{phrase_trigger}': {e}")
            phrase_token_ids = []
    
    vocab_size = len(tokenizer)
    processor = TagLogitProcessor(
        tag_token_id=tag_token_id,
        allowed_token_ids=allowed_token_ids,
        vocab_size=vocab_size,
        epsilon=epsilon,
        phrase_token_ids=phrase_token_ids,
        forced_token_id_if_phrase=forced_token_id_if_phrase,
    )
    
    print(f"TagLogitProcessor enabled: After <TAG>, only allow {allowed_tokens}")
    print(f"  Tag token ID: {tag_token_id}")
    print(f"  Allowed token IDs: {allowed_token_ids}")
    if phrase_token_ids and forced_token_id_if_phrase is not None:
        print(f"  Phrase trigger '{phrase_trigger}' detected -> force token id {forced_token_id_if_phrase}")
    
    return processor

