"""
Core utilities for model loading, generation, and evaluation
"""

import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from config import *

# Try to import BitsAndBytesConfig, but make it optional
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Warning: bitsandbytes not available. Will load models in full precision.")


def load_model_and_tokenizer(model_config):
    """Load model and tokenizer with optional 4-bit quantization"""
    print(f"Loading model: {model_config['model_name']}")
    
    # Configure 4-bit quantization if available
    bnb_config = None
    load_in_4bit = model_config.get('load_in_4bit', False)
    
    if load_in_4bit and BITSANDBYTES_AVAILABLE:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print("  Using 4-bit quantization")
        except Exception as e:
            print(f"  Warning: Could not configure 4-bit quantization: {e}")
            print("  Loading in full precision instead")
            bnb_config = None
    elif load_in_4bit and not BITSANDBYTES_AVAILABLE:
        print("  Warning: 4-bit quantization requested but bitsandbytes not available")
        print("  Loading in full precision instead")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['model_name'],
        trust_remote_code=True
    )
    
    # Prepare model loading kwargs
    model_kwargs = {
        'device_map': model_config.get('device_map', 'auto'),
        'trust_remote_code': True,
        'dtype': torch.float16,
    }
    
    # Add max_memory if specified
    if model_config.get('max_memory'):
        model_kwargs['max_memory'] = model_config['max_memory']
        print(f"  Limiting GPU memory: {model_config['max_memory']}")
    
    # Add offload_folder if specified
    if model_config.get('offload_folder'):
        model_kwargs['offload_folder'] = model_config['offload_folder']
        print(f"  Using offload folder: {model_config['offload_folder']}")
    
    # Only add quantization_config if we successfully created it
    if bnb_config is not None:
        model_kwargs['quantization_config'] = bnb_config
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config['model_name'],
        token = "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",
        **model_kwargs
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    return model, tokenizer


def load_gsm8k():
    """Load GSM8K test set"""
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    return dataset


def load_svamp():
    """Load SVAMP dataset"""
    print("Loading SVAMP dataset...")
    dataset = load_dataset("ChilleD/SVAMP", split="test")
    return dataset


def extract_answer(text):
    """
    Extract numerical answer from model output.
    Looks for patterns like "####", "The answer is", etc.
    """
    # GSM8K format: #### followed by number
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Look for "The answer is X" pattern
    match = re.search(r'(?:the answer is|answer:|final answer:)\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', 
                     text.lower())
    if match:
        return match.group(1).replace(',', '')
    
    # Look for boxed answer
    match = re.search(r'\\boxed\{(-?\d+(?:,\d{3})*(?:\.\d+)?)\}', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Last resort: find last number in text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return None


def normalize_answer(answer):
    """Normalize answer string for comparison"""
    if answer is None:
        return None
    
    # Remove commas and convert to float for comparison
    try:
        return float(str(answer).replace(',', ''))
    except (ValueError, AttributeError):
        return None


def check_answer(predicted, ground_truth):
    """Check if predicted answer matches ground truth"""
    pred_normalized = normalize_answer(predicted)
    gt_normalized = normalize_answer(ground_truth)
    
    if pred_normalized is None or gt_normalized is None:
        return False
    
    # Use small epsilon for float comparison
    return abs(pred_normalized - gt_normalized) < 1e-4


def create_prompt(question, dataset_format='gsm8k'):
    """Create prompt for the model"""
    if dataset_format == 'gsm8k':
        prompt = f"""Solve this math problem step by step. Show your work and end your answer with #### followed by just the numerical answer.

Question: {question}

Solution:"""
    else:
        prompt = f"""Solve this math problem step by step. Show your work.

Question: {question}

Solution:"""
    
    return prompt


def generate_solution(model, tokenizer, prompt, max_new_tokens, generation_config):
    """Generate solution with specified token budget"""
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Update generation config with max_new_tokens
    gen_config = generation_config.copy()
    gen_config['max_new_tokens'] = max_new_tokens
    gen_config['pad_token_id'] = tokenizer.pad_token_id
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_config
        )
    
    # Decode (remove prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text


def evaluate_predictions(results):
    """Calculate accuracy and other metrics from results"""
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def save_results(results, filepath):
    """Save results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_stats(results):
    """Print summary statistics"""
    metrics = evaluate_predictions(results)
    print(f"\nResults Summary:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Correct: {metrics['correct']}/{metrics['total']}")
    
    # Token statistics
    tokens_used = [r['tokens_used'] for r in results]
    print(f"  Avg tokens: {np.mean(tokens_used):.1f}")
    print(f"  Token std: {np.std(tokens_used):.1f}")