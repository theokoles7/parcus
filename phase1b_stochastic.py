"""
Phase 1b: Stochastic Verification

Generate multiple samples at saturation budgets to verify that plateau is real,
not just sampling noise.
"""

import torch
import random
import numpy as np
from tqdm import tqdm
import argparse

from config import *
from utils import *


def run_stochastic_verification(model_name, budgets, num_samples_per_problem=5, 
                                dataset_name='gsm8k', num_problems=None, seed=RANDOM_SEED):
    """
    Generate multiple samples per problem to verify saturation
    
    Args:
        model_name: Model identifier
        budgets: Budgets to test (typically saturation budgets like [512, 1024])
        num_samples_per_problem: Number of generations per problem
        dataset_name: Dataset to use
        num_problems: Number of problems to test (None = all)
        seed: Random seed
    """
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
    
    if num_problems is not None:
        dataset = dataset.select(range(min(num_problems, len(dataset))))
    
    print(f"\n{'='*60}")
    print(f"Running Phase 1b: Stochastic Verification")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name} ({len(dataset)} problems)")
    print(f"  Budgets: {budgets}")
    print(f"  Samples per problem: {num_samples_per_problem}")
    print(f"{'='*60}\n")
    
    all_results = {}
    
    for budget in budgets:
        print(f"\nTesting budget: {budget} tokens with {num_samples_per_problem} samples each")
        
        results = []
        
        for idx, example in enumerate(tqdm(dataset, desc=f"Budget {budget}")):
            question = example['question']
            ground_truth = example[answer_key]
            
            if '####' in str(ground_truth):
                ground_truth = ground_truth.split('####')[-1].strip()
            
            prompt = create_prompt(question, dataset_name)
            
            # Generate multiple samples
            samples = []
            for sample_idx in range(num_samples_per_problem):
                # Set different seed for each sample
                torch.manual_seed(seed + idx * 1000 + sample_idx)
                
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
                    
                    samples.append({
                        'sample_id': sample_idx,
                        'generated': generated,
                        'predicted': str(predicted) if predicted else None,
                        'correct': correct,
                        'tokens_used': tokens_used
                    })
                    
                except Exception as e:
                    print(f"\nError on problem {idx}, sample {sample_idx}: {e}")
                    samples.append({
                        'sample_id': sample_idx,
                        'generated': None,
                        'predicted': None,
                        'correct': False,
                        'tokens_used': 0,
                        'error': str(e)
                    })
            
            # Aggregate statistics
            correct_count = sum(1 for s in samples if s['correct'])
            avg_tokens = np.mean([s['tokens_used'] for s in samples])
            std_tokens = np.std([s['tokens_used'] for s in samples])
            
            results.append({
                'problem_id': idx,
                'question': question,
                'ground_truth': str(ground_truth),
                'samples': samples,
                'budget': budget,
                'correct_count': correct_count,
                'success_rate': correct_count / num_samples_per_problem,
                'avg_tokens': avg_tokens,
                'std_tokens': std_tokens
            })
        
        # Save results
        output_path = get_output_path('phase1b_stochastic', model_name, budget)
        save_results(results, output_path)
        
        # Print statistics
        print(f"\nResults for budget {budget}:")
        avg_success_rate = np.mean([r['success_rate'] for r in results])
        std_success_rate = np.std([r['success_rate'] for r in results])
        
        print(f"  Mean success rate: {avg_success_rate:.2%} ± {std_success_rate:.2%}")
        print(f"  Problems with 100% success: {sum(1 for r in results if r['success_rate'] == 1.0)}")
        print(f"  Problems with 0% success: {sum(1 for r in results if r['success_rate'] == 0.0)}")
        print(f"  Mean token usage: {np.mean([r['avg_tokens'] for r in results]):.1f}")
        
        all_results[budget] = results
    
    # Save combined results
    combined_path = get_output_path('phase1b_combined', model_name)
    save_results(all_results, combined_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("Stochastic Verification Summary:")
    print(f"{'='*60}")
    print(f"{'Budget':<10} {'Avg Success':<15} {'Std Dev':<15}")
    print(f"{'-'*40}")
    
    for budget in budgets:
        success_rates = [r['success_rate'] for r in all_results[budget]]
        print(f"{budget:<10} {np.mean(success_rates):<15.2%} {np.std(success_rates):<15.2%}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Phase 1b: Stochastic Verification')
    parser.add_argument('--model', type=str, default=PRIMARY_MODEL,
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to use')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                       choices=['gsm8k', 'svamp'],
                       help='Dataset to use')
    parser.add_argument('--budgets', type=int, nargs='+', default=STOCHASTIC_BUDGETS,
                       help='Token budgets to test')
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES_STOCHASTIC,
                       help='Number of samples per problem')
    parser.add_argument('--num-problems', type=int, default=None,
                       help='Number of problems to test (default: all)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed')
    
    args = parser.parse_args()
    
    run_stochastic_verification(
        model_name=args.model,
        budgets=args.budgets,
        num_samples_per_problem=args.samples,
        dataset_name=args.dataset,
        num_problems=args.num_problems,
        seed=args.seed
    )