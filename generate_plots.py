"""
Generate figures for ACL 2025 paper:
"When More Tokens Hurt: Saturation Effects in Test-Time Compute Scaling"

Usage: python generate_paper_figures.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# ADD THESE LINES:
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # For math symbols to match Times

# Model configuration
models = {
    'qwen-0.5b': 'Qwen 0.5B',
    'qwen-1.5b': 'Qwen 1.5B', 
    'qwen-3b': 'Qwen 3B',
    'qwen-7b': 'Qwen 7B'
}

# Load all combined phase1 results
results = {}
for model_key in models.keys():
    filepath = f'results/phase1_combined_{model_key}.json'
    with open(filepath, 'r') as f:
        results[model_key] = json.load(f)

# Calculate accuracy for each budget
budgets = [32, 64, 128, 256, 512, 1024, 2048]
accuracy_data = {}

for model_key in models.keys():
    accuracy_data[model_key] = {}
    for budget in budgets:
        budget_results = results[model_key].get(str(budget), [])
        if budget_results:
            correct = sum(1 for r in budget_results if r.get('correct', False))
            total = len(budget_results)
            accuracy_data[model_key][budget] = correct / total if total > 0 else 0.0
        else:
            accuracy_data[model_key][budget] = 0.0

print("Data loaded successfully.")
print(f"Total experiments: {len(models)} models × {len(budgets)} budgets × 1000 problems = {len(models) * len(budgets) * 1000}")

# =============================================================================
# FIGURE 1: Accuracy vs Budget Comparison
# =============================================================================

print("\nGenerating Figure 1: Accuracy vs Budget Comparison...")

fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

for (model_key, model_name), color in zip(models.items(), colors):
    accuracies = [accuracy_data[model_key][b] for b in budgets]
    ax.plot(budgets, accuracies, 'o-', label=model_name, linewidth=3, 
            markersize=10, color=color)

ax.set_xlabel('Max New Tokens (Budget)', fontsize=24)
ax.set_ylabel('Accuracy', fontsize=24)
# ax.set_title('Token Budget Scaling Across Model Sizes\n(GSM8K Dataset)', 
            #  fontsize=18, fontweight='bold')
ax.set_xscale('log', base=2)
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.legend(fontsize=20, loc='upper left', framealpha=0.9)
ax.tick_params(axis = "both", which = "major", labelsize = 20)
ax.set_ylim([0, 0.95])

# Add saturation markers (optional - shows where each model plateaus)
for model_key, color in zip(models.keys(), colors):
    accuracies = [accuracy_data[model_key][b] for b in budgets]
    max_acc = max(accuracies)
    sat_idx = next(i for i, acc in enumerate(accuracies) if acc >= max_acc - 0.01)
    ax.axhline(y=max_acc, color=color, linestyle='--', alpha=0.3, linewidth=2)
    ax.axvline(x=budgets[sat_idx], color=color, linestyle='--', alpha=0.3, linewidth=2)

plt.tight_layout()
plt.savefig('accuracy_vs_budget_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: accuracy_vs_budget_comparison.png")

# =============================================================================
# FIGURE 2: Parameter Efficiency
# =============================================================================

print("\nGenerating Figure 2: Parameter Efficiency...")

fig, ax = plt.subplots(figsize=(10, 6))

param_counts = [0.5, 1.5, 3.0, 7.0]
budget_levels = [256, 512, 1024]
budget_colors = ['#e67e22', '#16a085', '#8e44ad']

x = np.arange(len(param_counts))
width = 0.25

for i, (budget, color) in enumerate(zip(budget_levels, budget_colors)):
    effs = [accuracy_data[mk][budget] / p for mk, p in zip(models.keys(), param_counts)]
    ax.bar(x + i*width, effs, width, label=f'{budget} tokens', color=color, alpha=0.8)

ax.set_xlabel('Model Size (Billion Parameters)', fontsize=20)
ax.set_ylabel('Accuracy per Billion Parameters', fontsize=20)
# ax.set_title('Parameter Efficiency at Different Token Budgets', fontsize=16, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(['0.5B', '1.5B', '3B', '7B'], fontsize=16)
ax.legend(fontsize=20, loc='upper right')
ax.tick_params(axis = "y", which = "major", labelsize = 16)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('parameter_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: parameter_efficiency.png")

# =============================================================================
# Print summary statistics
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

for model_key, model_name in models.items():
    accuracies = [accuracy_data[model_key][b] for b in budgets]
    max_acc = max(accuracies)
    max_idx = accuracies.index(max_acc)
    
    # Find biggest jump
    max_jump = 0
    max_jump_from = None
    max_jump_to = None
    for i in range(len(budgets) - 1):
        jump = accuracies[i+1] - accuracies[i]
        if jump > max_jump:
            max_jump = jump
            max_jump_from = budgets[i]
            max_jump_to = budgets[i+1]
    
    print(f"\n{model_name}:")
    print(f"  Peak accuracy: {max_acc*100:.1f}% at {budgets[max_idx]} tokens")
    print(f"  Biggest jump: +{max_jump*100:.1f}% ({max_jump_from}→{max_jump_to} tokens)")
    print(f"  Accuracy at 2048: {accuracies[-1]*100:.1f}%")
    if accuracies[-1] < max_acc:
        print(f"  Degradation: -{(max_acc - accuracies[-1])*100:.1f}%")

print("\n" + "=" * 80)
print("Figures generated successfully!")
print("=" * 80)