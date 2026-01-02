#!/usr/bin/env python3
"""Prepare routing training data from routeguard dataset."""

import json
import random
from pathlib import Path


def load_and_convert_data(input_dir: str, output_train: str, output_val: str, val_ratio: float = 0.1):
    """
    Load JSON files from input_dir, merge them, convert to routing format, and split into train/val.
    
    Args:
        input_dir: Directory containing input JSON files
        output_train: Path to output training JSON file
        output_val: Path to output validation JSON file
        val_ratio: Ratio of data to use for validation (default 0.1 = 10%)
    """
    input_path = Path(input_dir)
    all_data = []
    
    # Load all JSON files
    for json_file in input_path.glob("*.json"):
        print(f"Loading {json_file.name}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"  Found {len(data)} items")
            all_data.extend(data)
    
    print(f"\nTotal items loaded: {len(all_data)}")
    
    # Convert to routing format
    converted_data = []
    for item in all_data:
        # Extract user input and model output from the input field
        input_text = item.get("input", "")
        
        # The input format is: "User: <user_input>\nModel: <model_output>"
        # We need to extract both parts
        if "User:" in input_text and "Model:" in input_text:
            # Split by "Model:" to get user and model parts
            parts = input_text.split("Model:", 1)
            user_part = parts[0].replace("User:", "").strip()
            model_part = parts[1].strip() if len(parts) > 1 else ""
            
            # Create the routing prompt
            routing_prompt = (
                "You are a routing classification system designed to analyze user inputs and model outputs to determine the appropriate guardrail model for safety evaluation.\n\n"
                "Your task is to classify each input-output pair and route it to either \"model 1\" or \"model 2\" based on the content characteristics.\n\n"
                f"User Input: {user_part}\n"
                f"Model Output: {model_part}\n\n"
                "Classification:"
            )
            
            # The answer is the output from the original data
            answer = item.get("output", "")
            
            converted_data.append({
                "prompt": routing_prompt,
                "answer": answer
            })
        else:
            print(f"Warning: Skipping item with unexpected input format: {input_text[:100]}...")
    
    print(f"Converted {len(converted_data)} items")
    
    # Shuffle data
    random.seed(42)
    random.shuffle(converted_data)
    
    # Split into train and validation
    val_size = int(len(converted_data) * val_ratio)
    val_data = converted_data[:val_size]
    train_data = converted_data[val_size:]
    
    print(f"\nSplit:")
    print(f"  Training: {len(train_data)} items")
    print(f"  Validation: {len(val_data)} items")
    
    # Save training data
    output_train_path = Path(output_train)
    output_train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved training data to: {output_train_path}")
    
    # Save validation data
    output_val_path = Path(output_val)
    output_val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"Saved validation data to: {output_val_path}")


if __name__ == "__main__":
    input_dir = "/data/wenjie_jacky_mo/data/routeguard/train"
    output_train = "/data/wenjie_jacky_mo/EasyR1/data/routing/train_routing_prompt_answer.json"
    output_val = "/data/wenjie_jacky_mo/EasyR1/data/routing/val_routing_prompt_answer.json"
    
    load_and_convert_data(input_dir, output_train, output_val, val_ratio=0.1)

