"""
Improved Automated Error Classification
"""

import pandas as pd
import re
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import *


def extract_numbers(text):
    """Extract all numbers from text"""
    if not text or pd.isna(text):
        return []
    # Match integers and decimals, including negatives
    return re.findall(r'-?\d+\.?\d*', str(text))


def classify_error_improved(row):
    """
    Improved automatic classification
    
    Priority:
    1. Check if solution is actually incomplete
    2. Check for arithmetic errors
    3. Check for setup/logic errors based on answer distance
    4. Default classification
    """
    question = str(row.get('question', ''))
    ground_truth = str(row.get('ground_truth', ''))
    generated = str(row.get('generated', ''))
    predicted = str(row.get('predicted', ''))
    tokens_used = row.get('tokens_used', 0)
    budget = row.get('budget', 0)
    
    # 1. TRUE INCOMPLETE: No answer extracted at all
    if not predicted or predicted == 'None' or predicted == 'nan' or str(predicted).strip() == '':
        # Check if solution looks cut off (doesn't end with period or ####)
        if generated and not re.search(r'[.!?#]{2,}\s*$', generated.strip()):
            return 'incomplete', 'generation cut off mid-sentence', 'yes', 0.9
        else:
            return 'format', 'answer not extracted from complete solution', 'yes', 0.85
    
    # 2. Check if used very few tokens (actually incomplete)
    if tokens_used < budget * 0.5:  # Used less than half the budget
        return 'incomplete', 'stopped early', 'yes', 0.85
    
    # 3. ARITHMETIC ERRORS: Check for wrong calculations
    # Look for explicit calculations like "5 * 8 = 40"
    arithmetic_error_found = False
    calc_patterns = [
        r'(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # addition
        r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # subtraction
        r'(\d+\.?\d*)\s*\*\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # multiplication
        r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',  # division
    ]
    
    for pattern in calc_patterns:
        calculations = re.findall(pattern, generated)
        for calc in calculations:
            try:
                num1, num2, result = float(calc[0]), float(calc[1]), float(calc[2])
                
                # Determine operation from pattern
                if '+' in pattern:
                    correct = num1 + num2
                elif '-' in pattern:
                    correct = num1 - num2
                elif r'\*' in pattern or '×' in pattern:
                    correct = num1 * num2
                elif '/' in pattern or '÷' in pattern:
                    correct = num1 / num2 if num2 != 0 else None
                else:
                    continue
                
                if correct is not None and abs(correct - result) > 0.1:
                    arithmetic_error_found = True
                    return 'arithmetic', f'wrong calc: {num1}±{num2}≠{result}', 'yes', 0.9
            except (ValueError, TypeError, ZeroDivisionError):
                continue
    
    # 4. SETUP vs LOGIC vs ARITHMETIC based on answer distance
    try:
        pred_val = float(predicted)
        gt_val = float(ground_truth)
        
        if gt_val == 0:
            if pred_val != 0:
                return 'setup', 'predicted non-zero for zero answer', 'no', 0.8
        else:
            error_ratio = abs(pred_val - gt_val) / abs(gt_val)
            
            # Very close (within 10%) - likely arithmetic
            if error_ratio < 0.1:
                return 'arithmetic', f'close but wrong ({error_ratio:.1%} off)', 'yes', 0.75
            
            # Moderately wrong (10-100%) - likely logic error
            elif error_ratio < 1.0:
                return 'logic', f'reasoning error ({error_ratio:.1%} off)', 'no', 0.7
            
            # Way off (>100%) - likely setup error
            else:
                return 'setup', f'fundamental error ({error_ratio:.1%} off)', 'no', 0.75
                
    except (ValueError, TypeError):
        # Predicted is not a number - check if answer is in text
        if ground_truth.strip() in generated:
            return 'format', 'answer in text but not extracted', 'yes', 0.8
        else:
            return 'logic', 'non-numeric prediction', 'no', 0.6
    
    # 5. Check for hallucination (too many new numbers)
    question_numbers = set(extract_numbers(question))
    solution_numbers = set(extract_numbers(generated))
    new_numbers = solution_numbers - question_numbers
    
    if len(new_numbers) > 10 and len(new_numbers) > 2 * len(question_numbers):
        return 'hallucination', 'many invented numbers', 'no', 0.65
    
    # 6. Default fallback
    return 'logic', 'unknown error type', 'no', 0.5


def auto_annotate_file(input_path, output_path=None):
    """Automatically annotate a TSV file with improved logic"""
    if output_path is None:
        output_path = input_path
    
    print(f"\nAuto-annotating: {input_path}")
    
    df = pd.read_csv(input_path, sep='\t')
    
    # Apply classification
    annotations = df.apply(classify_error_improved, axis=1)
    
    # Unpack results
    df['error_type'] = [a[0] for a in annotations]
    df['error_location'] = [a[1] for a in annotations]
    df['recoverable'] = [a[2] for a in annotations]
    df['notes'] = [f'auto (conf: {a[3]:.0%})' for a in annotations]
    
    # Save
    df.to_csv(output_path, sep='\t', index=False)
    
    # Print summary
    print(f"  Classified {len(df)} errors:")
    error_counts = df['error_type'].value_counts()
    for error_type, count in error_counts.items():
        pct = count / len(df) * 100
        print(f"    {error_type:15s}: {count:3d} ({pct:5.1f}%)")
    
    recoverable = (df['recoverable'] == 'yes').sum()
    unrecoverable = (df['recoverable'] == 'no').sum()
    print(f"  Recoverable: {recoverable}, Unrecoverable: {unrecoverable}")
    
    return df


def auto_annotate_all():
    """Auto-annotate all error files"""
    
    print("\n" + "="*70)
    print("AUTO-ANNOTATING ERROR FILES (Improved Logic)")
    print("="*70)
    
    annotated_files = []
    
    for budget in ANNOTATION_SAMPLES.keys():
        error_file = get_annotation_path(budget, 'errors')
        
        if error_file.exists():
            df = auto_annotate_file(error_file)
            annotated_files.append((budget, df))
        else:
            print(f"\nWarning: {error_file} not found, skipping")
    
    if not annotated_files:
        print("\nNo annotation files found!")
        return
    
    print("\n" + "="*70)
    print("SUMMARY ACROSS ALL BUDGETS")
    print("="*70)
    
    all_errors = pd.concat([df for _, df in annotated_files])
    
    print(f"\nTotal errors annotated: {len(all_errors)}")
    
    print(f"\nError type distribution:")
    error_dist = all_errors['error_type'].value_counts()
    for error_type, count in error_dist.items():
        pct = count / len(all_errors) * 100
        print(f"  {error_type:15s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nRecoverability:")
    recov_dist = all_errors['recoverable'].value_counts()
    for status, count in recov_dist.items():
        pct = count / len(all_errors) * 100
        print(f"  {status:15s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Run: python view_annotations.py --budget 512 --sample 10")
    print("     (to spot-check annotations)")
    print("  2. Run: python phase2_analyze.py")
    print("     (to generate analysis)")
    print("  3. Run: python visualize.py")
    print("     (to regenerate figures)")
    print("="*70 + "\n")
    
    return annotated_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-annotate error files')
    parser.add_argument('--file', type=str, help='Single file to annotate')
    parser.add_argument('--all', action='store_true', help='Annotate all error files')
    
    args = parser.parse_args()
    
    if args.file:
        auto_annotate_file(args.file)
    elif args.all:
        auto_annotate_all()
    else:
        # Default: annotate all
        auto_annotate_all()