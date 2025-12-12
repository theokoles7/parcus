"""
Phase 3b: Solution Structure Analysis

Analyze how solutions differ across budgets (setup, computation, verification phases)
"""

import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

from config import *
from utils import load_results


def analyze_solution_structure(text):
    """
    Analyze what tokens are used for (heuristic)
    
    Returns:
        Dict with counts for each phase
    """
    if not text:
        return {'setup': 0, 'computation': 0, 'verification': 0, 'other': 0}
    
    tokens = text.lower().split()
    
    phases = {
        'setup': 0,
        'computation': 0,
        'verification': 0,
        'other': 0
    }
    
    # Count tokens in each phase
    for token in tokens:
        classified = False
        
        # Check setup keywords
        for keyword in SOLUTION_PHASES['setup']:
            if keyword in token:
                phases['setup'] += 1
                classified = True
                break
        
        if classified:
            continue
        
        # Check computation keywords
        for keyword in SOLUTION_PHASES['computation']:
            if keyword in token:
                phases['computation'] += 1
                classified = True
                break
        
        if classified:
            continue
        
        # Check verification keywords
        for keyword in SOLUTION_PHASES['verification']:
            if keyword in token:
                phases['verification'] += 1
                classified = True
                break
        
        if not classified:
            phases['other'] += 1
    
    return phases


def count_arithmetic_operations(text):
    """Count arithmetic operations in the solution"""
    if not text:
        return 0
    
    # Look for arithmetic operators
    operations = len(re.findall(r'[+\-*/=]', text))
    
    # Look for words indicating computation
    computation_words = ['add', 'subtract', 'multiply', 'divide', 'equals', 'sum', 'difference', 'product']
    for word in computation_words:
        operations += text.lower().count(word)
    
    return operations


def count_reasoning_steps(text):
    """Estimate number of reasoning steps"""
    if not text:
        return 0
    
    # Count sentences/steps
    sentences = len(re.findall(r'[.!?]+', text))
    
    # Count step markers
    step_markers = len(re.findall(r'(?:step|first|second|third|next|then|finally)', text.lower()))
    
    return max(sentences, step_markers)


def detect_verification(text):
    """Check if solution includes verification"""
    if not text:
        return False
    
    verification_patterns = [
        r'check',
        r'verify',
        r'confirm',
        r'indeed',
        r'correct',
        r'let.*check',
        r'double.?check'
    ]
    
    text_lower = text.lower()
    for pattern in verification_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def compare_across_budgets(results_by_budget, problem_id):
    """
    Compare solution structure for same problem across budgets
    """
    
    comparison = {}
    
    for budget, results in sorted(results_by_budget.items()):
        # Find this problem
        problem_result = next((r for r in results if r['problem_id'] == problem_id), None)
        
        if problem_result and problem_result.get('generated'):
            text = problem_result['generated']
            
            comparison[budget] = {
                'correct': problem_result['correct'],
                'tokens_used': problem_result['tokens_used'],
                'structure': analyze_solution_structure(text),
                'operations': count_arithmetic_operations(text),
                'steps': count_reasoning_steps(text),
                'has_verification': detect_verification(text)
            }
    
    return comparison


def analyze_structure_evolution(model_name, budgets):
    """
    Analyze how solution structure evolves with budget
    """
    
    print(f"\n{'='*60}")
    print(f"Solution Structure Analysis")
    print(f"  Model: {model_name}")
    print(f"  Budgets: {budgets}")
    print(f"{'='*60}\n")
    
    # Load results
    results_by_budget = {}
    for budget in budgets:
        # Try phase3 first, fall back to phase1
        path = get_output_path('phase3_cliff', model_name, budget)
        if not path.exists():
            path = get_output_path('phase1_main', model_name, budget)
        
        if path.exists():
            results_by_budget[budget] = load_results(path)
            print(f"Loaded results for budget {budget}")
        else:
            print(f"Warning: No results found for budget {budget}")
    
    if not results_by_budget:
        print("No results found! Run phase1 or phase3 first.")
        return
    
    # Aggregate statistics
    stats_by_budget = {}
    
    for budget, results in results_by_budget.items():
        structures = []
        operations = []
        steps = []
        verifications = []
        
        for r in results:
            if r.get('generated'):
                text = r['generated']
                structures.append(analyze_solution_structure(text))
                operations.append(count_arithmetic_operations(text))
                steps.append(count_reasoning_steps(text))
                verifications.append(detect_verification(text))
        
        # Aggregate
        avg_structure = {
            'setup': np.mean([s['setup'] for s in structures]),
            'computation': np.mean([s['computation'] for s in structures]),
            'verification': np.mean([s['verification'] for s in structures]),
            'other': np.mean([s['other'] for s in structures])
        }
        
        stats_by_budget[budget] = {
            'avg_structure': avg_structure,
            'avg_operations': np.mean(operations),
            'avg_steps': np.mean(steps),
            'pct_with_verification': np.mean(verifications),
            'accuracy': sum(r['correct'] for r in results) / len(results)
        }
    
    # Print summary
    print(f"\n{'Budget':<10} {'Accuracy':<12} {'Steps':<10} {'Operations':<12} {'% Verify':<12}")
    print(f"{'-'*56}")
    
    for budget in sorted(stats_by_budget.keys()):
        stats = stats_by_budget[budget]
        print(f"{budget:<10} {stats['accuracy']:<12.2%} {stats['avg_steps']:<10.1f} "
              f"{stats['avg_operations']:<12.1f} {stats['pct_with_verification']:<12.1%}")
    
    # Detailed structure
    print(f"\n{'Budget':<10} {'Setup':<10} {'Compute':<12} {'Verify':<12} {'Other':<10}")
    print(f"{'-'*54}")
    
    for budget in sorted(stats_by_budget.keys()):
        struct = stats_by_budget[budget]['avg_structure']
        print(f"{budget:<10} {struct['setup']:<10.1f} {struct['computation']:<12.1f} "
              f"{struct['verification']:<12.1f} {struct['other']:<10.1f}")
    
    # Find problems that transition at cliff
    print(f"\n\nAnalyzing problems that transition at the cliff...")
    
    # Focus on 128 vs 256 transition
    if 128 in results_by_budget and 256 in results_by_budget:
        results_128 = {r['problem_id']: r for r in results_by_budget[128]}
        results_256 = {r['problem_id']: r for r in results_by_budget[256]}
        
        # Find problems that fail at 128 but succeed at 256
        transitions = []
        for pid in results_128.keys():
            if pid in results_256:
                if not results_128[pid]['correct'] and results_256[pid]['correct']:
                    transitions.append(pid)
        
        print(f"  Found {len(transitions)} problems that transition from fail → success")
        
        if transitions:
            # Analyze a few examples
            print(f"\n  Example transitions:")
            for pid in transitions[:3]:
                comparison = compare_across_budgets(
                    {128: results_by_budget[128], 256: results_by_budget[256]},
                    pid
                )
                
                print(f"\n    Problem {pid}:")
                for budget, data in sorted(comparison.items()):
                    print(f"      {budget} tokens: {data['steps']} steps, "
                          f"{data['operations']} ops, verify={data['has_verification']}")
    
    # Save analysis
    output = {
        'budgets': budgets,
        'stats_by_budget': {str(k): v for k, v in stats_by_budget.items()}
    }
    
    output_path = RESULTS_DIR / f'phase3_structure_analysis_{model_name}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Analysis saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return stats_by_budget


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze solution structure across budgets')
    parser.add_argument('--model', type=str, default=PRIMARY_MODEL,
                       help='Model name')
    parser.add_argument('--budgets', type=int, nargs='+', 
                       default=[128, 144, 160, 176, 192, 208, 224, 240, 256],
                       help='Budgets to analyze')
    
    args = parser.parse_args()
    
    analyze_structure_evolution(args.model, args.budgets)