"""
Configuration for saturation study experiments
"""

import os
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Create directories
for dir_path in [DATA_DIR, RESULTS_DIR, ANNOTATIONS_DIR, FIGURES_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ============================================================================
# Model Configuration
# ============================================================================

# GPU Memory Management
# If your display freezes during experiments, try these settings:
# 1. Set MAX_MEMORY to limit GPU usage (e.g., {'0': '10GB'} to leave 6GB for display)
# 2. Set OFFLOAD_FOLDER to use CPU RAM as backup
# 3. Enable SLEEP_BETWEEN_PROBLEMS to give GPU breaks
MAX_MEMORY = None  # e.g., {'0': '10GB'} to limit to 10GB on GPU 0
OFFLOAD_FOLDER = None  # e.g., './offload' to use disk for overflow
SLEEP_BETWEEN_PROBLEMS = 0  # Seconds to sleep between problems (e.g., 0.1)

MODEL_CONFIGS = {
    'qwen-1.5b': {
        'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
        'load_in_4bit': True,  # Set to False if bitsandbytes not available
        'device_map': 'auto',
        'max_memory': MAX_MEMORY,
        'offload_folder': OFFLOAD_FOLDER,
    },
    'qwen-7b': {
        'model_name': 'Qwen/Qwen2.5-7B-Instruct',
        'load_in_4bit': True,  # Set to False if bitsandbytes not available
        'device_map': 'auto',
        'max_memory': MAX_MEMORY,
        'offload_folder': OFFLOAD_FOLDER,
    },
    'llama-3-1b': {
        'model_name': 'meta-llama/Llama-3.2-1B-Instruct',
        'load_in_4bit': True,  # Set to False if bitsandbytes not available
        'device_map': 'auto',
        'max_memory': MAX_MEMORY,
        'offload_folder': OFFLOAD_FOLDER,
    }
}

# NOTE: If you get bitsandbytes errors, either:
# 1. Install bitsandbytes: pip install bitsandbytes
# 2. Or set 'load_in_4bit': False above (will use more GPU memory)

# Primary model for main experiments
PRIMARY_MODEL = 'qwen-1.5b'

# Secondary model for comparison (set to None if not testing)
SECONDARY_MODEL = 'qwen-7b'  # or None

# ============================================================================
# Dataset Configuration
# ============================================================================
DATASETS = {
    'gsm8k': {
        'name': 'gsm8k',
        'split': 'test',
        'primary': True,
    },
    'svamp': {
        'name': 'ChilleD/SVAMP',
        'split': 'test',
        'primary': False,
    }
}

# ============================================================================
# Experiment Configuration
# ============================================================================

# Phase 1: Main saturation curve
MAIN_BUDGETS = [32, 64, 128, 256, 512, 1024, 2048]

# Phase 1b: Stochastic verification
STOCHASTIC_BUDGETS = [512, 1024]
NUM_SAMPLES_STOCHASTIC = 5

# Phase 3: Fine-grained cliff analysis
CLIFF_BUDGETS = [96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]

# Generation parameters
GENERATION_CONFIG = {
    'temperature': 0.7,
    'top_p': 0.9,
    'do_sample': True,
    'pad_token_id': None,  # Will be set to eos_token_id
}

# ============================================================================
# Annotation Configuration
# ============================================================================

ERROR_TYPES = {
    'arithmetic': 'Computational error (e.g., 2+2=5, 15*3=40)',
    'setup': 'Misunderstood problem or set up wrong equation',
    'logic': 'Reasoning step doesn\'t follow (non-sequitur)',
    'incomplete': 'Stopped before reaching answer',
    'format': 'Correct reasoning but wrong output format',
    'hallucination': 'Introduced facts not in problem',
    'none': 'Correct answer'
}

# Annotation sampling strategy
ANNOTATION_SAMPLES = {
    128: 30,   # errors to annotate at budget 128
    256: 30,   # errors to annotate at budget 256
    512: 50,   # errors to annotate at budget 512
    1024: 50,  # errors to annotate at budget 1024
}

ANNOTATION_SUCCESSES = 20  # successes to annotate per budget

# ============================================================================
# Analysis Configuration
# ============================================================================

# Solution structure keywords
SOLUTION_PHASES = {
    'setup': ['let', 'given', 'we have', 'suppose', 'define', 'assume'],
    'computation': ['therefore', 'so', 'thus', 'equals', 'calculate', 'compute'],
    'verification': ['check', 'verify', 'confirm', 'indeed', 'correct', 'validate']
}

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Utility Functions
# ============================================================================

def get_output_path(experiment_name, model_name, budget=None, ext='json'):
    """Generate standardized output path"""
    filename = f"{experiment_name}_{model_name}"
    if budget is not None:
        filename += f"_budget{budget}"
    filename += f".{ext}"
    return RESULTS_DIR / filename

def get_annotation_path(budget, annotation_type='errors'):
    """Generate annotation file path"""
    return ANNOTATIONS_DIR / f"annotate_{annotation_type}_budget{budget}.tsv"