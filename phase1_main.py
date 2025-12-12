"""
Phase 1: Main Saturation Curve Experiment

Runs the primary model across all main budgets to establish the saturation phenomenon.
"""

import torch
import random
import numpy as np
from tqdm import tqdm
import argparse
import time

from config import *
from utils import *


def run_experiment(model_name, budgets, dataset_name='gsm8k', num_samples=None, seed=RANDOM_SEED):
    """
    Run main saturation experiment
    
    Args:
        model_name: Model identifier from MODEL_CONFIGS
        budgets: List of token budgets to test
        dataset_name: Dataset to use ('gsm8k' or 'svamp')
        num_samples: Number of problems to test (None = all)
        seed: Random seed for reproducibility
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
    
    # Limit dataset size if specified
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"\n{'='*60}")
    print(f"Running Phase 1: Main Saturation Curve")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name} ({len(dataset)} problems)")
    print(f"  Budgets: {budgets}")
    print(f"{'='*60}\n")
    
    # Run experiments for each budget
    all_results = {}
    
    for budget in budgets:
        print(f"\nTesting budget: {budget} tokens")
        
        results = []
        
        for idx, example in enumerate(tqdm(dataset, desc=f"Budget {budget}")):
            question = example['question']
            ground_truth = example[answer_key]
            
            # Extract numerical answer from ground truth if needed
            if '####' in str(ground_truth):
                ground_truth = ground_truth.split('####')[-1].strip()
            
            # Create prompt
            prompt = create_prompt(question, dataset_name)
            
            # Generate solution
            try:
                generated = generate_solution(
                    model, 
                    tokenizer, 
                    prompt, 
                    budget,
                    GENERATION_CONFIG
                )
                
                # Extract answer
                predicted = extract_answer(generated)
                
                # Check correctness
                correct = check_answer(predicted, ground_truth)
                
                # Count tokens used
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
            
            # Sleep between problems if configured (helps prevent display freezing)
            if SLEEP_BETWEEN_PROBLEMS > 0:
                time.sleep(SLEEP_BETWEEN_PROBLEMS)
        
        # Save results for this budget
        output_path = get_output_path('phase1_main', model_name, budget)
        save_results(results, output_path)
        
        # Print statistics
        print_stats(results)
        
        all_results[budget] = results
    
    # Save combined results
    combined_path = get_output_path('phase1_combined', model_name)
    save_results(all_results, combined_path)
    
    # Print summary across budgets
    print(f"\n{'='*60}")
    print("Summary Across Budgets:")
    print(f"{'='*60}")
    print(f"{'Budget':<10} {'Accuracy':<12} {'Avg Tokens':<12}")
    print(f"{'-'*34}")
    
    for budget in budgets:
        metrics = evaluate_predictions(all_results[budget])
        avg_tokens = np.mean([r['tokens_used'] for r in all_results[budget]])
        print(f"{budget:<10} {metrics['accuracy']:<12.2%} {avg_tokens:<12.1f}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Phase 1: Main Saturation Curve')
    parser.add_argument('--model', type=str, default=PRIMARY_MODEL,
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to use')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                       choices=['gsm8k', 'svamp'],
                       help='Dataset to use')
    parser.add_argument('--budgets', type=int, nargs='+', default=MAIN_BUDGETS,
                       help='Token budgets to test')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of problems to test (default: all)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed')
    
    args = parser.parse_args()
    
    run_experiment(
        model_name=args.model,
        budgets=args.budgets,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        seed=args.seed
    )