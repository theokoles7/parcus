"""
Phase 2: Error Annotation System

Create annotation files and provide interface for error taxonomy.
"""

import json
import random
import pandas as pd
from pathlib import Path
import argparse

from config import *
from utils import load_results, evaluate_predictions


def sample_for_annotation(results, n_errors, n_successes, seed=RANDOM_SEED):
    """
    Sample errors and successes for annotation
    
    Args:
        results: List of result dictionaries
        n_errors: Number of errors to sample
        n_successes: Number of successes to sample
        seed: Random seed
    
    Returns:
        (errors, successes): Tuple of sampled lists
    """
    random.seed(seed)
    
    errors = [r for r in results if not r['correct']]
    successes = [r for r in results if r['correct']]
    
    # Sample (with replacement if necessary)
    sampled_errors = random.sample(errors, min(n_errors, len(errors)))
    sampled_successes = random.sample(successes, min(n_successes, len(successes)))
    
    return sampled_errors, sampled_successes


def create_annotation_file(samples, output_path, annotation_type='error'):
    """
    Create TSV file for annotation
    
    Args:
        samples: List of problem results
        output_path: Path to save annotation file
        annotation_type: 'error' or 'success'
    """
    
    rows = []
    for sample in samples:
        rows.append({
            'problem_id': sample['problem_id'],
            'budget': sample['budget'],
            'question': sample['question'],
            'ground_truth': sample['ground_truth'],
            'generated': sample['generated'][:1000] + '...' if len(sample.get('generated', '')) > 1000 else sample.get('generated', ''),
            'predicted': sample.get('predicted', 'N/A'),
            'correct': sample['correct'],
            'tokens_used': sample['tokens_used'],
            'error_type': '',  # To be filled in
            'error_location': '',  # To be filled in (e.g., "step 3", "setup", etc.)
            'recoverable': '',  # To be filled in (yes/no/unknown)
            'notes': ''  # Additional notes
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Created annotation file: {output_path}")
    print(f"  {len(rows)} samples")
    
    return df


def prepare_all_annotations(model_name='qwen-1.5b', dataset_name='gsm8k'):
    """
    Prepare all annotation files based on ANNOTATION_SAMPLES config
    """
    
    print(f"\n{'='*60}")
    print(f"Preparing Annotation Files")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}\n")
    
    annotation_files = []
    
    for budget, n_errors in ANNOTATION_SAMPLES.items():
        # Load results for this budget
        results_path = get_output_path('phase1_main', model_name, budget)
        
        if not results_path.exists():
            print(f"Warning: Results not found for budget {budget}: {results_path}")
            print(f"  Run phase1_main.py first!")
            continue
        
        print(f"\nProcessing budget {budget}:")
        results = load_results(results_path)
        
        # Print statistics
        metrics = evaluate_predictions(results)
        print(f"  Total: {metrics['total']}, Correct: {metrics['correct']}, Errors: {metrics['total'] - metrics['correct']}")
        
        # Sample errors
        errors, successes = sample_for_annotation(
            results, 
            n_errors, 
            ANNOTATION_SUCCESSES,
            seed=RANDOM_SEED + budget
        )
        
        print(f"  Sampled: {len(errors)} errors, {len(successes)} successes")
        
        # Create annotation files
        error_path = get_annotation_path(budget, 'errors')
        success_path = get_annotation_path(budget, 'successes')
        
        create_annotation_file(errors, error_path, 'error')
        create_annotation_file(successes, success_path, 'success')
        
        annotation_files.append({
            'budget': budget,
            'error_file': str(error_path),
            'success_file': str(success_path),
            'n_errors': len(errors),
            'n_successes': len(successes)
        })
    
    # Save annotation manifest
    manifest_path = ANNOTATIONS_DIR / 'annotation_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({
            'model': model_name,
            'dataset': dataset_name,
            'error_types': ERROR_TYPES,
            'files': annotation_files
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Annotation files created!")
    print(f"{'='*60}")
    print(f"\nManifest saved to: {manifest_path}")
    print(f"\nAnnotation instructions:")
    print(f"  1. Open each TSV file in a spreadsheet editor")
    print(f"  2. For each row, fill in:")
    print(f"     - error_type: One of {list(ERROR_TYPES.keys())}")
    print(f"     - error_location: Where the error occurred (e.g., 'step 2', 'setup')")
    print(f"     - recoverable: Could more tokens have fixed this? (yes/no/unknown)")
    print(f"     - notes: Any additional observations")
    print(f"  3. Save the files")
    print(f"  4. Run phase2_analyze.py to analyze annotations")
    
    return annotation_files


def validate_annotations(annotation_file):
    """
    Validate that annotations are complete and correct
    """
    df = pd.read_csv(annotation_file, sep='\t')
    
    issues = []
    
    # Check for missing error types
    missing_type = df['error_type'].isna() | (df['error_type'] == '')
    if missing_type.any():
        issues.append(f"{missing_type.sum()} rows missing error_type")
    
    # Check for invalid error types
    valid_types = set(ERROR_TYPES.keys())
    invalid = df[~df['error_type'].isin(valid_types) & ~missing_type]
    if len(invalid) > 0:
        issues.append(f"{len(invalid)} rows have invalid error_type")
        print(f"  Invalid types: {invalid['error_type'].unique()}")
    
    # Check recoverable field
    valid_recoverable = {'yes', 'no', 'unknown', ''}
    invalid_rec = df[~df['recoverable'].astype(str).str.lower().isin(valid_recoverable)]
    if len(invalid_rec) > 0:
        issues.append(f"{len(invalid_rec)} rows have invalid recoverable value")
    
    if issues:
        print(f"\nValidation issues in {annotation_file}:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✓ {annotation_file} is valid!")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare Phase 2 annotation files')
    parser.add_argument('--model', type=str, default=PRIMARY_MODEL,
                       help='Model name')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                       help='Dataset name')
    parser.add_argument('--validate', type=str, default=None,
                       help='Path to annotation file to validate')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_annotations(args.validate)
    else:
        prepare_all_annotations(args.model, args.dataset)