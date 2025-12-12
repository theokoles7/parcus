"""
Visualization utilities for creating publication-quality figures
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from config import *
from utils import load_results, evaluate_predictions


# Set style for publication-quality figures
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'


def plot_saturation_curve(results_by_budget, model_name='', save_path=None):
    """
    Plot main saturation curve (accuracy vs token budget)
    
    Args:
        results_by_budget: Dict mapping budget -> list of results
        model_name: Model name for title
        save_path: Path to save figure
    """
    
    budgets = sorted(results_by_budget.keys())
    accuracies = []
    std_errors = []
    
    for budget in budgets:
        results = results_by_budget[budget]
        metrics = evaluate_predictions(results)
        acc = metrics['accuracy']
        # Standard error for binomial
        n = metrics['total']
        se = np.sqrt(acc * (1 - acc) / n)
        
        accuracies.append(acc)
        std_errors.append(se)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with error bars
    ax.errorbar(budgets, accuracies, yerr=std_errors, 
                marker='o', linewidth=2, markersize=8,
                capsize=5, capthick=2, label='Accuracy')
    
    # Add saturation line if applicable
    if len(budgets) > 3:
        saturation_acc = accuracies[-1]
        ax.axhline(y=saturation_acc, color='r', linestyle='--', 
                  label=f'Saturation: {saturation_acc:.1%}', alpha=0.7)
    
    ax.set_xlabel('Token Budget', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(f'Performance Saturation Curve{" - " + model_name if model_name else ""}', 
                fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax


def plot_error_distribution(annotation_dfs, save_path=None):
    """
    Plot stacked bar chart of error types by budget
    
    Args:
        annotation_dfs: Dict mapping budget -> DataFrame of annotations
        save_path: Path to save figure
    """
    
    # Prepare data
    budgets = sorted(annotation_dfs.keys())
    error_types = list(ERROR_TYPES.keys())
    error_types.remove('none')  # Only plot actual errors
    
    data = {et: [] for et in error_types}
    
    for budget in budgets:
        df = annotation_dfs[budget]
        counts = df['error_type'].value_counts()
        total = len(df)
        
        for et in error_types:
            pct = counts.get(et, 0) / total if total > 0 else 0
            data[et].append(pct)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(budgets))
    width = 0.6
    
    bottom = np.zeros(len(budgets))
    colors = sns.color_palette('Set2', len(error_types))
    
    for i, et in enumerate(error_types):
        ax.bar(x, data[et], width, label=et.capitalize(), 
              bottom=bottom, color=colors[i])
        bottom += data[et]
    
    ax.set_xlabel('Token Budget', fontsize=14)
    ax.set_ylabel('Proportion of Errors', fontsize=14)
    ax.set_title('Error Type Distribution by Budget', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.legend(fontsize=10, loc='upper right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax


def plot_cliff_detail(results_by_budget, save_path=None):
    """
    Plot fine-grained accuracy around the cliff region
    
    Args:
        results_by_budget: Dict mapping budget -> list of results
        save_path: Path to save figure
    """
    
    budgets = sorted(results_by_budget.keys())
    accuracies = []
    
    for budget in budgets:
        results = results_by_budget[budget]
        metrics = evaluate_predictions(results)
        accuracies.append(metrics['accuracy'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot accuracy curve
    ax.plot(budgets, accuracies, marker='o', linewidth=2, markersize=6)
    
    # Compute and plot derivative (gain per token)
    if len(budgets) > 1:
        gains = []
        midpoints = []
        for i in range(len(budgets) - 1):
            gain = (accuracies[i+1] - accuracies[i]) / (budgets[i+1] - budgets[i])
            gains.append(gain * 100)  # Scale for visibility
            midpoints.append((budgets[i] + budgets[i+1]) / 2)
        
        ax2 = ax.twinx()
        ax2.plot(midpoints, gains, 'r--', marker='s', linewidth=2, 
                markersize=5, alpha=0.7, label='Gain per token (×100)')
        ax2.set_ylabel('Accuracy Gain per Token (×100)', fontsize=14, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right', fontsize=10)
    
    ax.set_xlabel('Token Budget', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Fine-Grained Analysis of the Cliff Region', fontsize=16)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax


def plot_solution_structure(stats_by_budget, save_path=None):
    """
    Plot solution structure breakdown across budgets
    
    Args:
        stats_by_budget: Dict mapping budget -> structure statistics
        save_path: Path to save figure
    """
    
    budgets = sorted(stats_by_budget.keys())
    
    # Extract data
    setup = [stats_by_budget[b]['avg_structure']['setup'] for b in budgets]
    computation = [stats_by_budget[b]['avg_structure']['computation'] for b in budgets]
    verification = [stats_by_budget[b]['avg_structure']['verification'] for b in budgets]
    other = [stats_by_budget[b]['avg_structure']['other'] for b in budgets]
    
    # Create stacked area plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.stackplot(budgets, setup, computation, verification, other,
                labels=['Setup', 'Computation', 'Verification', 'Other'],
                alpha=0.8, colors=sns.color_palette('Set2', 4))
    
    ax.set_xlabel('Token Budget', fontsize=14)
    ax.set_ylabel('Average Token Count', fontsize=14)
    ax.set_title('Solution Structure by Budget', fontsize=16)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax


def plot_verification_vs_accuracy(stats_by_budget, save_path=None):
    """
    Plot relationship between verification and accuracy
    
    Args:
        stats_by_budget: Dict mapping budget -> structure statistics
        save_path: Path to save figure
    """
    
    budgets = sorted(stats_by_budget.keys())
    
    verification_pct = [stats_by_budget[b]['pct_with_verification'] for b in budgets]
    accuracies = [stats_by_budget[b]['accuracy'] for b in budgets]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot with budget labels
    scatter = ax.scatter(verification_pct, accuracies, s=100, alpha=0.6)
    
    # Add budget labels
    for i, budget in enumerate(budgets):
        ax.annotate(str(budget), (verification_pct[i], accuracies[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('% Solutions with Verification', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Verification vs Accuracy', fontsize=16)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig, ax


def create_all_figures(model_name='qwen-1.5b'):
    """
    Generate all publication figures
    """
    
    print(f"\n{'='*60}")
    print("Generating All Figures")
    print(f"{'='*60}\n")
    
    figures_created = []
    
    # Figure 1: Main saturation curve
    print("Creating Figure 1: Saturation Curve...")
    try:
        combined_path = get_output_path('phase1_combined', model_name)
        if combined_path.exists():
            import json
            with open(combined_path) as f:
                results_by_budget = {int(k): v for k, v in json.load(f).items()}
            
            fig_path = FIGURES_DIR / f'fig1_saturation_curve_{model_name}.png'
            plot_saturation_curve(results_by_budget, model_name, fig_path)
            figures_created.append(fig_path)
        else:
            print(f"  Skipped: No phase1 results found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Figure 2: Error distribution
    print("\nCreating Figure 2: Error Distribution...")
    try:
        annotation_dfs = {}
        for budget in ANNOTATION_SAMPLES.keys():
            path = get_annotation_path(budget, 'errors')
            if path.exists():
                annotation_dfs[budget] = pd.read_csv(path, sep='\t')
        
        if annotation_dfs:
            fig_path = FIGURES_DIR / f'fig2_error_distribution_{model_name}.png'
            plot_error_distribution(annotation_dfs, fig_path)
            figures_created.append(fig_path)
        else:
            print(f"  Skipped: No annotations found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Figure 3: Cliff detail
    print("\nCreating Figure 3: Cliff Detail...")
    try:
        combined_path = get_output_path('phase3_combined', model_name)
        if combined_path.exists():
            import json
            with open(combined_path) as f:
                results_by_budget = {int(k): v for k, v in json.load(f).items()}
            
            fig_path = FIGURES_DIR / f'fig3_cliff_detail_{model_name}.png'
            plot_cliff_detail(results_by_budget, fig_path)
            figures_created.append(fig_path)
        else:
            print(f"  Skipped: No phase3 results found")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Figure 4: Solution structure
    print("\nCreating Figure 4: Solution Structure...")
    try:
        structure_path = RESULTS_DIR / f'phase3_structure_analysis_{model_name}.json'
        if structure_path.exists():
            import json
            with open(structure_path) as f:
                data = json.load(f)
                stats_by_budget = {int(k): v for k, v in data['stats_by_budget'].items()}
            
            fig_path = FIGURES_DIR / f'fig4_solution_structure_{model_name}.png'
            plot_solution_structure(stats_by_budget, fig_path)
            figures_created.append(fig_path)
            
            # Bonus: verification vs accuracy
            fig_path = FIGURES_DIR / f'fig5_verification_accuracy_{model_name}.png'
            plot_verification_vs_accuracy(stats_by_budget, fig_path)
            figures_created.append(fig_path)
        else:
            print(f"  Skipped: No structure analysis found")
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Created {len(figures_created)} figures:")
    for path in figures_created:
        print(f"  - {path}")
    print(f"{'='*60}\n")
    
    return figures_created


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create all visualization figures')
    parser.add_argument('--model', type=str, default=PRIMARY_MODEL,
                       help='Model name')
    
    args = parser.parse_args()
    
    create_all_figures(args.model)