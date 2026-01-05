"""
Annotation Viewer - Spot-check auto-annotations
"""

import pandas as pd
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def view_annotation(row, show_full=False):
    """Pretty-print a single annotation"""
    
    print("\n" + "="*80)
    print(f"Problem ID: {row['problem_id']} | Budget: {row['budget']} tokens")
    print("="*80)
    
    # Question
    print(f"\n📝 QUESTION:")
    question = row['question']
    if len(question) > 200 and not show_full:
        print(f"   {question[:200]}...")
    else:
        print(f"   {question}")
    
    # Ground truth
    print(f"\n✓ CORRECT ANSWER: {row['ground_truth']}")
    
    # Model's solution
    print(f"\n🤖 MODEL'S SOLUTION ({row['tokens_used']} tokens used):")
    generated = str(row['generated'])
    if len(generated) > 400 and not show_full:
        print(f"   {generated[:400]}...")
        print(f"   [...{len(generated)-400} more characters...]")
    else:
        print(f"   {generated}")
    
    # Model's predicted answer
    print(f"\n❌ PREDICTED ANSWER: {row['predicted']}")
    
    # Auto-annotation
    print(f"\n🏷️  AUTO-ANNOTATION:")
    print(f"   Error Type:  {row['error_type']}")
    print(f"   Location:    {row['error_location']}")
    print(f"   Recoverable: {row['recoverable']}")
    print(f"   Notes:       {row['notes']}")
    
    print("\n" + "="*80)


def view_sample(budget, n=10, error_type=None, show_full=False, random_sample=True):
    """View a sample of annotations"""
    
    annotation_file = get_annotation_path(budget, 'errors')
    
    if not annotation_file.exists():
        print(f"❌ Error: {annotation_file} not found")
        print(f"   Run auto_annotate_improved.py first!")
        return
    
    print(f"\n{'='*80}")
    print(f"VIEWING ANNOTATIONS: Budget {budget}")
    print(f"{'='*80}")
    
    df = pd.read_csv(annotation_file, sep='\t')
    
    # Filter by error type if specified
    if error_type:
        df = df[df['error_type'] == error_type]
        print(f"Filtered to error_type='{error_type}': {len(df)} errors")
    
    if len(df) == 0:
        print("No errors found matching criteria")
        return
    
    # Sample
    if random_sample:
        sample_df = df.sample(min(n, len(df)))
    else:
        sample_df = df.head(n)
    
    print(f"Showing {len(sample_df)} of {len(df)} total errors\n")
    
    for idx, row in sample_df.iterrows():
        view_annotation(row, show_full=show_full)
        
        if idx < len(sample_df) - 1:
            input("\nPress Enter to see next annotation (or Ctrl+C to quit)...")


def view_by_id(budget, problem_id):
    """View a specific problem by ID"""
    
    annotation_file = get_annotation_path(budget, 'errors')
    
    if not annotation_file.exists():
        print(f"❌ Error: {annotation_file} not found")
        return
    
    df = pd.read_csv(annotation_file, sep='\t')
    
    matching = df[df['problem_id'] == problem_id]
    
    if len(matching) == 0:
        print(f"❌ No annotation found for problem_id={problem_id} at budget={budget}")
        return
    
    view_annotation(matching.iloc[0], show_full=True)


def show_summary(budget=None):
    """Show summary statistics"""
    
    if budget:
        budgets = [budget]
    else:
        budgets = ANNOTATION_SAMPLES.keys()
    
    print("\n" + "="*80)
    print("ANNOTATION SUMMARY")
    print("="*80)
    
    for b in budgets:
        annotation_file = get_annotation_path(b, 'errors')
        
        if not annotation_file.exists():
            print(f"\nBudget {b}: No annotations found")
            continue
        
        df = pd.read_csv(annotation_file, sep='\t')
        
        print(f"\n📊 Budget {b} ({len(df)} errors):")
        print(f"   Error types:")
        for error_type, count in df['error_type'].value_counts().items():
            pct = count / len(df) * 100
            print(f"      {error_type:15s}: {count:3d} ({pct:5.1f}%)")
        
        recoverable = (df['recoverable'] == 'yes').sum()
        unrecoverable = (df['recoverable'] == 'no').sum()
        print(f"   Recoverable: {recoverable} ({recoverable/len(df)*100:.1f}%)")
        print(f"   Unrecoverable: {unrecoverable} ({unrecoverable/len(df)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='View and spot-check auto-annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View 10 random errors from budget 512
  python view_annotations.py --budget 512 --sample 10
  
  # View only arithmetic errors from budget 256
  python view_annotations.py --budget 256 --type arithmetic --sample 5
  
  # View a specific problem
  python view_annotations.py --budget 512 --id 58
  
  # Show summary of all budgets
  python view_annotations.py --summary
  
  # View first 5 errors (not random)
  python view_annotations.py --budget 128 --sample 5 --no-random
        """
    )
    
    parser.add_argument('--budget', type=int, help='Budget to view')
    parser.add_argument('--sample', type=int, default=10, help='Number of samples to view')
    parser.add_argument('--type', type=str, help='Filter by error type')
    parser.add_argument('--id', type=int, help='View specific problem ID')
    parser.add_argument('--full', action='store_true', help='Show full text (no truncation)')
    parser.add_argument('--no-random', action='store_true', help='Show first N instead of random')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics')
    
    args = parser.parse_args()
    
    if args.summary:
        show_summary(args.budget)
    elif args.id is not None:
        if not args.budget:
            print("Error: --budget required when using --id")
        else:
            view_by_id(args.budget, args.id)
    elif args.budget:
        view_sample(
            args.budget, 
            n=args.sample, 
            error_type=args.type,
            show_full=args.full,
            random_sample=not args.no_random
        )
    else:
        parser.print_help()