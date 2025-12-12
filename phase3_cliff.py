"""
Phase 3: Fine-Grained Cliff Analysis

Test budgets around the 128-256 token cliff to understand the transition.
"""

import torch
import random
import numpy as np
from tqdm import tqdm
import argparse

from config import *
from utils import *


def run_cliff_analysis(model_name, budgets=CLIFF_BUDGETS, dataset_name='gsm8k', 
                      num_samples=None, seed=RANDOM_SEED):
    """
    Run fine-grained analysis around the cliff region
    
    Args:
        model_name: Model identifier
        budgets: Fine-grained budget values around the cliff
        dataset_name: Dataset to use
        num_samples: Number of problems (None = all)
        seed: Random seed
    """
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load model
    model_config = MODEL_CONFIGS[model_name]
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # Load dataset
    if dataset_name == 'gsm8k':
        dataset = load_gsm8k()
        answer_key = 'answer'
    elif dataset_name == 'svamp':
        dataset = load_svamp()
        answer_key = 'Answer'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"\n{'='*60}")
    print(f"Running Phase 3: Fine-Grained Cliff Analysis")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name} ({len(dataset)} problems)")
    print(f"  Budgets: {budgets}")
    print(f"{'='*60}\n")
    
    # Run experiments
    all_results = {}
    
    for budget in budgets:
        print(f"\nTesting budget: {budget} tokens")
        
        results = []
        
        for idx, example in enumerate(tqdm(dataset, desc=f"Budget {budget}")):
            question = example['question']
            ground_truth = example[answer_key]
            
            if '####' in str(ground_truth):
                ground_truth = ground_truth.split('####')[-1].strip()
            
            prompt = create_prompt(question, dataset_name)
            
            try:
                generated = generate_solution(
                    model, 
                    tokenizer, 
                    prompt, 
                    budget,
                    GENERATION_CONFIG
                )
                
                predicted = extract_answer(generated)
                correct = check_answer(predicted, ground_truth)
                tokens_used = len(tokenizer.encode(generated))
                
                results.append({
                    'problem_id': idx,
                    'question': question,
                    'ground_truth': str(ground_truth),
                    'generated': generated,
                    'predicted': str(predicted) if predicted else None,
                    'correct': correct,
                    'tokens_used': tokens_used,
                    'budget': budget
                })
                
            except Exception as e:
                print(f"\nError on problem {idx}: {e}")
                results.append({
                    'problem_id': idx,
                    'question': question,
                    'ground_truth': str(ground_truth),
                    'generated': None,
                    'predicted': None,
                    'correct': False,
                    'tokens_used': 0,
                    'budget': budget,
                    'error': str(e)
                })
        
        # Save results
        output_path = get_output_path('phase3_cliff', model_name, budget)
        save_results(results, output_path)
        print_stats(results)
        
        all_results[budget] = results
    
    # Save combined
    combined_path = get_output_path('phase3_combined', model_name)
    save_results(all_results, combined_path)
    
    # Analyze cliff
    print(f"\n{'='*60}")
    print("Cliff Analysis Summary:")
    print(f"{'='*60}")
    
    # Calculate accuracy gains
    print(f"\n{'Budget':<10} {'Accuracy':<12} {'Gain/Token':<15}")
    print(f"{'-'*37}")
    
    prev_budget = None
    prev_acc = None
    
    for budget in budgets:
        metrics = evaluate_predictions(all_results[budget])
        acc = metrics['accuracy']
        
        if prev_budget is not None:
            gain_per_token = (acc - prev_acc) / (budget - prev_budget)
            print(f"{budget:<10} {acc:<12.2%} {gain_per_token:<15.4f}")
        else:
            print(f"{budget:<10} {acc:<12.2%} {'-':<15}")
        
        prev_budget = budget
        prev_acc = acc
    
    # Identify cliff region (largest gain/token)
    gains = []
    for i in range(1, len(budgets)):
        b1, b2 = budgets[i-1], budgets[i]
        acc1 = evaluate_predictions(all_results[b1])['accuracy']
        acc2 = evaluate_predictions(all_results[b2])['accuracy']
        gain = (acc2 - acc1) / (b2 - b1)
        gains.append((b1, b2, gain))
    
    gains.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nSteepest improvements:")
    for b1, b2, gain in gains[:3]:
        print(f"  {b1} → {b2}: {gain:.4f} accuracy per token")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Phase 3: Cliff Analysis')
    parser.add_argument('--model', type=str, default=PRIMARY_MODEL,
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to use')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                       choices=['gsm8k', 'svamp'],
                       help='Dataset to use')
    parser.add_argument('--budgets', type=int, nargs='+', default=CLIFF_BUDGETS,
                       help='Token budgets to test')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of problems to test (default: all)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed')
    
    args = parser.parse_args()
    
    run_cliff_analysis(
        model_name=args.model,
        budgets=args.budgets,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        seed=args.seed
    )