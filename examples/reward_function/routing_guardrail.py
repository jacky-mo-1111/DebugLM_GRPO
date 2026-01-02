#!/usr/bin/env python3
"""Reward function for routing model that routes to guardrail models and checks correctness."""

from __future__ import annotations

import os
import re
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

try:
    import requests
except ImportError:
    requests = None


# Extract routing decision from response
ROUTING_PATTERN = re.compile(r"(?:model\s*)?([12])\b", re.IGNORECASE)


def _extract_routing_decision(response: str) -> int | None:
    """Extract model number (1 or 2) from routing model's response."""
    if not response:
        return None
    
    # Try to find "model 1" or "model 2" or just "1" or "2"
    match = ROUTING_PATTERN.search(response)
    if match:
        return int(match.group(1))
    
    return None


def _extract_user_input_and_model_output(prompt: str) -> tuple[str, str]:
    """Extract user input and model output from the routing prompt.
    
    The prompt format is:
    "You are a routing classification system designed to analyze user inputs and model outputs...\n\nUser Input: <user_input>\nModel Output: <model_output>\n\nClassification:"
    
    Returns:
        Tuple of (user_input, model_output)
    """
    # Try new format first: "User Input:" and "Model Output:"
    # Use more flexible pattern to handle various separators
    user_match = re.search(r"User Input:\s*(.+?)(?:\n\s*Model Output:|$)", prompt, re.DOTALL)
    model_match = re.search(r"Model Output:\s*(.+?)(?:\n\s*\n\s*Classification:|$)", prompt, re.DOTALL)
    
    if user_match and model_match:
        user_input = user_match.group(1).strip()
        model_output = model_match.group(1).strip()
        if user_input and model_output:
            return user_input, model_output
    
    # Fallback: try without strict separator matching
    user_match = re.search(r"User Input:\s*(.+?)(?=Model Output:|$)", prompt, re.DOTALL)
    model_match = re.search(r"Model Output:\s*(.+?)(?=Classification:|$)", prompt, re.DOTALL)
    
    if user_match and model_match:
        user_input = user_match.group(1).strip()
        model_output = model_match.group(1).strip()
        if user_input and model_output:
            return user_input, model_output
    
    # Fallback to old format: "User:" and "Model:"
    user_match = re.search(r"User:\s*(.+?)(?:\n\s*Model:|$)", prompt, re.DOTALL)
    model_match = re.search(r"Model:\s*(.+?)(?:\n|$)", prompt, re.DOTALL)
    
    user_input = user_match.group(1).strip() if user_match else ""
    model_output = model_match.group(1).strip() if model_match else ""
    
    return user_input, model_output


def _get_model_name(client: OpenAI) -> str | None:
    """Get the model name from vLLM API server.
    
    vLLM uses the model path as the model name, which can be retrieved from /v1/models endpoint.
    """
    # Try to get from /v1/models endpoint using requests
    if requests is not None:
        try:
            # Extract base URL (remove /v1 if present)
            base_url = str(client.base_url).rstrip('/v1').rstrip('/')
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    model_id = data["data"][0].get("id")
                    if model_id:
                        return model_id
        except Exception:
            pass
    
    # Try using OpenAI client's models.list() if available (newer OpenAI client versions)
    try:
        if hasattr(client, 'models') and hasattr(client.models, 'list'):
            models = client.models.list()
            if hasattr(models, 'data') and models.data and len(models.data) > 0:
                model_id = models.data[0].id if hasattr(models.data[0], 'id') else None
                if model_id:
                    return model_id
    except Exception:
        pass
    
    # Return None if we can't get the model name
    return None


def _call_guardrail_model(
    client: OpenAI,
    user_input: str,
    model_output: str,
    model_port: int,
    model_name: str | None = None,
) -> str:
    """Call a guardrail model via vLLM API and return its response.
    
    The guardrail model needs both user input and model output to determine safety.
    Note: The client should already be initialized with the correct base_url.
    """
    # Get model name if not provided
    if model_name is None:
        model_name = _get_model_name(client)
        if model_name is None:
            # If we can't get the model name, raise an error with helpful message
            raise ValueError(
                f"Could not retrieve model name from vLLM server at {client.base_url}. "
                "Please ensure the vLLM server is running and accessible. "
                "You can check by calling: curl http://localhost:{model_port}/v1/models"
            )
    
    # Construct the prompt for the guardrail model
    # The guardrail model should check if the model output is safe given the user input
    guardrail_prompt = f"User: {user_input}\nModel: {model_output}\n\nIs this response safe?"
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": guardrail_prompt}
            ],
            max_tokens=50,  # Guardrail models output short responses
            temperature=0.0,  # Deterministic output
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling guardrail model on port {model_port}: {e}")
        return ""


def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().lower()
    # Remove common variations
    answer = re.sub(r"^the\s+", "", answer)
    answer = re.sub(r"response\s+is\s+", "", answer)
    return answer


def compute_score(
    reward_inputs: list[dict[str, Any]],
    model1_port: int = 8000,
    model2_port: int = 8001,
    model1_base_url: str | None = None,
    model2_base_url: str | None = None,
) -> list[dict[str, float]]:
    """
    Compute reward for routing model.
    
    Args:
        reward_inputs: List of reward inputs, each containing:
            - "response": The routing model's output ("model 1" or "model 2")
            - "ground_truth": The expected answer ("The response is safe" or "The response is unsafe")
            - Additional fields may include the original prompt
        model1_port: Port for model 1 (child_abuse)
        model2_port: Port for model 2 (animal_abuse)
        model1_base_url: Optional base URL for model 1 (overrides port)
        model2_base_url: Optional base URL for model 2 (overrides port)
    
    Returns:
        List of reward scores, each containing:
            - "overall": 1.0 if routing was correct and guardrail model output matches ground truth, else 0.0
            - "routing_correct": 1.0 if routing decision was made, else 0.0
            - "guardrail_correct": 1.0 if guardrail model output matches ground truth, else 0.0
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for the routing reward function.")
    
    # Initialize OpenAI clients for both models
    client1 = OpenAI(
        base_url=model1_base_url or f"http://localhost:{model1_port}/v1",
        api_key="dummy",  # vLLM doesn't require real API key
    )
    client2 = OpenAI(
        base_url=model2_base_url or f"http://localhost:{model2_port}/v1",
        api_key="dummy",
    )
    
    # Get model names once for efficiency
    model1_name = _get_model_name(client1)
    model2_name = _get_model_name(client2)
    
    scores = []
    for reward_input in reward_inputs:
        routing_response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")
        
        # Extract routing decision
        routing_decision = _extract_routing_decision(routing_response)
        routing_correct = 1.0 if routing_decision is not None else 0.0
        
        if routing_decision is None:
            # If we can't parse routing decision, give 0 reward
            scores.append({
                "overall": 0.0,
                "routing_correct": 0.0,
                "guardrail_correct": 0.0,
            })
            continue
        
        # Extract user input from the original prompt
        # The prompt should be in reward_input, but we need to check where it is
        # It might be in a "prompt" field or we need to reconstruct it
        # For now, let's assume we can get it from the context
        # In practice, you might need to pass it differently
        
        # Try to get the original prompt
        # The prompt should be available in the reward_input, but the exact field name
        # depends on how the data is structured. Let's try common field names.
        original_prompt = (
            reward_input.get("prompt") or 
            reward_input.get("input") or 
            reward_input.get("query") or
            ""
        )
        
        if not original_prompt:
            # If prompt is not available, we can't call the guardrail model
            print("Warning: Original prompt not found in reward_input. Available keys:", reward_input.keys())
            scores.append({
                "overall": 0.0,
                "routing_correct": routing_correct,
                "guardrail_correct": 0.0,
            })
            continue
        
        # Extract user input and model output from the prompt
        user_input, model_output = _extract_user_input_and_model_output(original_prompt)
        
        if not user_input or not model_output:
            # Only print warning occasionally to avoid log spam
            import random
            if random.random() < 0.01:  # Print 1% of warnings
                print(f"Warning: Could not extract user input or model output from prompt")
                print(f"  Prompt preview: {original_prompt[:200]}...")
                print(f"  Extracted user_input: {user_input[:50] if user_input else 'None'}...")
                print(f"  Extracted model_output: {model_output[:50] if model_output else 'None'}...")
            scores.append({
                "overall": 0.0,
                "routing_correct": routing_correct,
                "guardrail_correct": 0.0,
            })
            continue
        
        # Call the appropriate guardrail model
        if routing_decision == 1:
            guardrail_response = _call_guardrail_model(
                client1, user_input, model_output, model1_port, model1_name
            )
        else:  # routing_decision == 2
            guardrail_response = _call_guardrail_model(
                client2, user_input, model_output, model2_port, model2_name
            )
        
        # Compare guardrail response with ground truth
        normalized_guardrail = _normalize_answer(guardrail_response)
        normalized_ground_truth = _normalize_answer(ground_truth)
        
        guardrail_correct = 1.0 if normalized_guardrail == normalized_ground_truth else 0.0
        
        # Overall reward: 1.0 only if routing was made AND guardrail output matches ground truth
        overall = 1.0 if (routing_correct > 0 and guardrail_correct > 0) else 0.0
        
        scores.append({
            "overall": overall,
            "routing_correct": routing_correct,
            "guardrail_correct": guardrail_correct,
        })
    
    return scores

