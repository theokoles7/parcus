#!/usr/bin/env python3
"""
Master runner script for saturation study

This script orchestrates all phases of the experiment.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import *


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error running: {description}")
        return False
    
    print(f"\n✓ Completed: {description}")
    return True


def run_phase1(args):
    """Run Phase 1: Main saturation curve"""
    cmd = [
        sys.executable, 'phase1_main.py',
        '--model', args.model,
        '--dataset', args.dataset,
    ]
    
    if args.num_samples:
        cmd.extend(['--num-samples', str(args.num_samples)])
    
    if not args.budgets:
        cmd.extend(['--budgets'] + [str(b) for b in MAIN_BUDGETS])
    else:
        cmd.extend(['--budgets'] + [str(b) for b in args.budgets])
    
    return run_command(cmd, "Phase 1: Main Saturation Curve")


def run_phase1b(args):
    """Run Phase 1b: Stochastic verification"""
    cmd = [
        sys.executable, 'phase1b_stochastic.py',
        '--model', args.model,
        '--dataset', args.dataset,
        '--samples', str(args.stochastic_samples),
    ]
    
    if args.num_samples:
        cmd.extend(['--num-problems', str(args.num_samples)])
    
    return run_command(cmd, "Phase 1b: Stochastic Verification")


def run_phase2_prepare(args):
    """Prepare Phase 2: Annotation files"""
    cmd = [
        sys.executable, 'phase2_prepare.py',
        '--model', args.model,
        '--dataset', args.dataset,
    ]
    
    success = run_command(cmd, "Phase 2: Prepare Annotations")
    
    if success:
        print(f"\n{'='*60}")
        print("IMPORTANT: Manual Annotation Required")
        print(f"{'='*60}")
        print("\nAnnotation files have been created in:")
        print(f"  {ANNOTATIONS_DIR}")
        print("\nPlease:")
        print("  1. Open each *_errors.tsv file")
        print("  2. Fill in the error_type, error_location, recoverable, and notes columns")
        print("  3. Save the files")
        print("  4. Run this script with --phase2-analyze to analyze annotations")
        print(f"\n{'='*60}\n")
    
    return success


def run_phase2_analyze(args):
    """Analyze Phase 2: Completed annotations"""
    cmd = [
        sys.executable, 'phase2_analyze.py',
        '--model', args.model,
    ]
    
    return run_command(cmd, "Phase 2: Analyze Annotations")


def run_phase3_cliff(args):
    """Run Phase 3: Cliff analysis"""
    cmd = [
        sys.executable, 'phase3_cliff.py',
        '--model', args.model,
        '--dataset', args.dataset,
    ]
    
    if args.num_samples:
        cmd.extend(['--num-samples', str(args.num_samples)])
    
    return run_command(cmd, "Phase 3: Fine-Grained Cliff Analysis")


def run_phase3_structure(args):
    """Run Phase 3b: Solution structure analysis"""
    cmd = [
        sys.executable, 'phase3_structure.py',
        '--model', args.model,
    ]
    
    return run_command(cmd, "Phase 3b: Solution Structure Analysis")


def run_visualize(args):
    """Generate all figures"""
    cmd = [
        sys.executable, 'visualize.py',
        '--model', args.model,
    ]
    
    return run_command(cmd, "Generating Figures")


def main():
    parser = argparse.ArgumentParser(
        description='Run saturation study experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything (except manual annotation)
  python run_all.py --all
  
  # Run just Phase 1
  python run_all.py --phase1
  
  # Run Phase 2 analysis (after manual annotation)
  python run_all.py --phase2-analyze
  
  # Run with limited samples for testing
  python run_all.py --phase1 --num-samples 100
  
  # Run on different model
  python run_all.py --all --model qwen-7b
        """
    )
    
    # Model and dataset
    parser.add_argument('--model', type=str, default=PRIMARY_MODEL,
                       help='Model to use')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                       choices=['gsm8k', 'svamp'],
                       help='Dataset to use')
    
    # Experiment selection
    parser.add_argument('--all', action='store_true',
                       help='Run all phases (except manual annotation)')
    parser.add_argument('--phase1', action='store_true',
                       help='Run Phase 1: Main saturation curve')
    parser.add_argument('--phase1b', action='store_true',
                       help='Run Phase 1b: Stochastic verification')
    parser.add_argument('--phase2-prepare', action='store_true',
                       help='Prepare Phase 2: Create annotation files')
    parser.add_argument('--phase2-analyze', action='store_true',
                       help='Analyze Phase 2: Analyze completed annotations')
    parser.add_argument('--phase3', action='store_true',
                       help='Run Phase 3: Cliff and structure analysis')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate all figures')
    
    # Parameters
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Limit number of problems (for testing)')
    parser.add_argument('--budgets', type=int, nargs='+',
                       help='Custom budget list for Phase 1')
    parser.add_argument('--stochastic-samples', type=int, default=NUM_SAMPLES_STOCHASTIC,
                       help='Number of samples per problem for stochastic verification')
    
    args = parser.parse_args()
    
    # If no phase specified, show help
    if not (args.all or args.phase1 or args.phase1b or args.phase2_prepare or 
            args.phase2_analyze or args.phase3 or args.visualize):
        parser.print_help()
        return
    
    # Create directories
    for dir_path in [DATA_DIR, RESULTS_DIR, ANNOTATIONS_DIR, FIGURES_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*60}")
    print("SATURATION STUDY - Experiment Runner")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    if args.num_samples:
        print(f"Limited to: {args.num_samples} samples")
    print(f"{'='*60}\n")
    
    # Track success
    all_success = True
    
    # Run experiments
    if args.all or args.phase1:
        all_success &= run_phase1(args)
    
    if args.all or args.phase1b:
        all_success &= run_phase1b(args)
    
    if args.all or args.phase2_prepare:
        all_success &= run_phase2_prepare(args)
    
    if args.phase2_analyze:
        all_success &= run_phase2_analyze(args)
    
    if args.all or args.phase3:
        all_success &= run_phase3_cliff(args)
        all_success &= run_phase3_structure(args)
    
    if args.all or args.visualize:
        all_success &= run_visualize(args)
    
    # Summary
    print(f"\n{'='*60}")
    if all_success:
        print("✓ All experiments completed successfully!")
    else:
        print("⚠ Some experiments failed. Check logs above.")
    print(f"{'='*60}\n")
    
    if args.all:
        print("Next steps:")
        print("  1. Complete manual annotations in:", ANNOTATIONS_DIR)
        print("  2. Run: python run_all.py --phase2-analyze")
        print("  3. Generate final figures: python run_all.py --visualize")


if __name__ == "__main__":
    main()