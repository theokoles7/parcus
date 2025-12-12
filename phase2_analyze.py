"""
Phase 2: Analyze Annotated Errors

Analyze the completed annotations to understand error patterns.
"""

import pandas as pd
import numpy as np
import json
from collections import Counter, defaultdict
import argparse

from config import *
from utils import load_results, evaluate_predictions


def load_annotations(budget, annotation_type='errors'):
    """Load annotation file"""
    path = get_annotation_path(budget, annotation_type)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    
    df = pd.read_csv(path, sep='\t')
    return df


def analyze_error_distribution(annotation_dfs):
    """
    Analyze distribution of error types across budgets
    
    Args:
        annotation_dfs: Dict mapping budget -> DataFrame of annotations
    
    Returns:
        Dict with analysis results
    """
    
    results = {}
    
    for budget, df in annotation_dfs.items():
        # Count error types
        error_counts = Counter(df['error_type'])
        total = len(df)
        
        # Calculate percentages
        error_pcts = {k: v/total for k, v in error_counts.items()}
        
        # Recoverable analysis
        recoverable_counts = Counter(df['recoverable'].str.lower())
        
        results[budget] = {
            'total': total,
            'error_counts': dict(error_counts),
            'error_percentages': error_pcts,
            'recoverable_counts': dict(recoverable_counts)
        }
    
    return results


def analyze_error_persistence(all_results):
    """
    Track which problems fail at multiple budgets with same error
    
    Args:
        all_results: Dict mapping budget -> list of results from phase1
    
    Returns:
        Analysis of persistent errors
    """
    # Group by problem_id
    problem_errors = defaultdict(list)
    
    for budget in sorted(all_results.keys()):
        results = all_results[budget]
        for r in results:
            if not r['correct']:
                problem_errors[r['problem_id']].append({
                    'budget': budget,
                    'predicted': r['predicted'],
                    'generated': r['generated']
                })
    
    # Classify persistence
    persistent = []
    recovered = []
    
    for problem_id, errors in problem_errors.items():
        if len(errors) >= 2:
            # Check if it's the same error repeated
            predictions = [e['predicted'] for e in errors]
            if len(set(predictions)) == 1:  # Same wrong answer
                persistent.append({
                    'problem_id': problem_id,
                    'num_budgets': len(errors),
                    'budgets': [e['budget'] for e in errors],
                    'predicted': predictions[0]
                })
        
        # Find problems that eventually succeeded
        budgets = sorted([e['budget'] for e in errors])
        if budgets[-1] < max(all_results.keys()):
            # Problem failed at some budgets but potentially succeeded later
            recovered.append({
                'problem_id': problem_id,
                'failed_budgets': budgets
            })
    
    return {
        'persistent_errors': persistent,
        'potentially_recovered': recovered,
        'num_persistent': len(persistent),
        'num_recovered': len(recovered)
    }


def create_error_matrix(annotation_dfs):
    """
    Create a matrix showing error type distribution by budget
    
    Returns:
        DataFrame with budgets as rows, error types as columns
    """
    
    budgets = sorted(annotation_dfs.keys())
    error_types = list(ERROR_TYPES.keys())
    
    matrix = []
    for budget in budgets:
        df = annotation_dfs[budget]
        counts = Counter(df['error_type'])
        total = len(df)
        
        row = {'budget': budget}
        for error_type in error_types:
            row[error_type] = counts.get(error_type, 0) / total if total > 0 else 0
        
        matrix.append(row)
    
    return pd.DataFrame(matrix)


def generate_report(annotation_dfs, all_results=None):
    """
    Generate comprehensive analysis report
    """
    
    print(f"\n{'='*60}")
    print("ERROR ANALYSIS REPORT")
    print(f"{'='*60}\n")
    
    # 1. Error distribution by budget
    print("1. ERROR DISTRIBUTION BY BUDGET")
    print("-" * 60)
    
    dist_analysis = analyze_error_distribution(annotation_dfs)
    
    for budget in sorted(annotation_dfs.keys()):
        print(f"\nBudget {budget} tokens:")
        data = dist_analysis[budget]
        print(f"  Total annotated: {data['total']}")
        print(f"  Error breakdown:")
        
        for error_type, count in sorted(data['error_counts'].items(), 
                                       key=lambda x: x[1], reverse=True):
            pct = data['error_percentages'][error_type]
            print(f"    {error_type:20s}: {count:3d} ({pct:6.1%})")
        
        print(f"  Recoverable analysis:")
        for recoverable, count in data['recoverable_counts'].items():
            print(f"    {recoverable:20s}: {count:3d}")
    
    # 2. Error matrix
    print(f"\n\n2. ERROR TYPE EVOLUTION MATRIX")
    print("-" * 60)
    
    matrix = create_error_matrix(annotation_dfs)
    print(matrix.to_string(index=False, float_format=lambda x: f"{x:.1%}"))
    
    # 3. Persistence analysis (if full results provided)
    if all_results:
        print(f"\n\n3. ERROR PERSISTENCE ANALYSIS")
        print("-" * 60)
        
        persistence = analyze_error_persistence(all_results)
        print(f"  Persistent errors (same wrong answer across budgets): {persistence['num_persistent']}")
        print(f"  Potentially recovered errors: {persistence['num_recovered']}")
        
        if persistence['persistent_errors']:
            print(f"\n  Top persistent errors:")
            for i, err in enumerate(persistence['persistent_errors'][:5], 1):
                print(f"    {i}. Problem {err['problem_id']}: failed at budgets {err['budgets']}")
    
    # 4. Recoverability summary
    print(f"\n\n4. RECOVERABILITY SUMMARY")
    print("-" * 60)
    
    all_recoverable = []
    for df in annotation_dfs.values():
        all_recoverable.extend(df['recoverable'].str.lower().tolist())
    
    rec_counts = Counter(all_recoverable)
    total_annotated = len(all_recoverable)
    
    print(f"  Across all annotations:")
    for status in ['yes', 'no', 'unknown']:
        count = rec_counts.get(status, 0)
        pct = count / total_annotated if total_annotated > 0 else 0
        print(f"    {status.capitalize():10s}: {count:3d} ({pct:6.1%})")
    
    # Save summary
    summary = {
        'error_distribution': dist_analysis,
        'error_matrix': matrix.to_dict('records'),
        'recoverability_summary': dict(rec_counts)
    }
    
    if all_results:
        summary['persistence'] = persistence
    
    summary_path = RESULTS_DIR / 'phase2_error_analysis.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Analysis saved to: {summary_path}")
    print(f"{'='*60}\n")
    
    return summary


def main(model_name='qwen-1.5b'):
    """Run full analysis"""
    
    # Load all annotation files
    annotation_dfs = {}
    
    print("Loading annotation files...")
    for budget in ANNOTATION_SAMPLES.keys():
        try:
            df = load_annotations(budget, 'errors')
            annotation_dfs[budget] = df
            print(f"  Budget {budget}: {len(df)} annotations")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
    
    if not annotation_dfs:
        print("\nNo annotation files found!")
        print("Run phase2_prepare.py first, then complete the annotations.")
        return
    
    # Load original results if available (for persistence analysis)
    all_results = {}
    for budget in annotation_dfs.keys():
        try:
            results_path = get_output_path('phase1_main', model_name, budget)
            all_results[budget] = load_results(results_path)
        except:
            pass
    
    # Generate report
    summary = generate_report(annotation_dfs, all_results if all_results else None)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Phase 2 annotations')
    parser.add_argument('--model', type=str, default=PRIMARY_MODEL,
                       help='Model name')
    
    args = parser.parse_args()
    
    main(args.model)