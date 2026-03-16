"""# parcus.commands.infer.main

Main process entry point for infer command.
"""

__all__ = ["infer_entry_point"]

from typing                         import Any, Dict, List, Optional

from parcus.commands.infer.__args__ import InferConfig
from parcus.registration            import register_command

@register_command(
    id =        "infer",
    config =    InferConfig
)
def infer_entry_point(
    model_id:       str,
    dataset_id:     str,
    token_budget:   Optional[List[int]] =   None,
    output_path:    str =                   "output",
    seed:           int =                   1,
    *args,
    **kwargs
) -> None:
    """# Conduct Model Inference on Dataset.

    ## Args:
        * model_id    (str):                Model family identifier (e.g. "qwen").
        * dataset_id    (str):              Dataset identifier (e.g. "gsm8k").
        * token_budget  (List[int] | None): List of token budgets to evaluate. None for 
                                            unconstrained.
        * output_path   (str):              Root path for results. Defaults to "output".
        * seed          (int):              Random number generation seed. Defaults to 1.
    """
    from json                   import dump
    from logging                import Logger
    from pathlib                import Path

    from tqdm                   import tqdm

    from parcus.datasets        import Dataset
    from parcus.models          import Model
    from parcus.registration    import DATASET_REGISTRY, MODEL_REGISTRY
    from parcus.utilities       import get_logger, set_hugging_face_token, set_seed

    # Initialize logger.
    logger: Logger =    get_logger("infer-process")

    # Set seed for reproducibility.
    set_seed(seed = seed)

    # Set hugging face token for improved API access.
    set_hugging_face_token(token_path = ".hf_token")

    # Load model.
    model:      Model =     MODEL_REGISTRY.load_model(model_id = model_id, **kwargs)

    # Load dataset.
    dataset:    Dataset =   DATASET_REGISTRY.load_dataset(
                                dataset_id = dataset_id,
                                **kwargs
                            )

    # Resolve output directory.
    output_dir: Path =      Path(output_path) / model.path.split("/")[1] /  \
                            dataset.id / dataset.subset / dataset._split_

    # Normalize budgets: None → [None] (unconstrained), else the provided list.
    budgets:    List[int] = token_budget

    # For each token budget...
    for budget in budgets:

        # Log configuration.
        logger.info(
            f"Infer process initiated ("
            f"model = {model_id}, "
            f"dataset = {dataset_id}, "
            f"budget(s) = {budget}, "
            f"seed = {seed})"
        )

        # Derive budget label for logging and file naming.
        budget_label:   str =                       str(budget)             \
                                                    if budget is not None   \
                                                    else "unconstrained"

        # Initialize sample results.
        samples:        Dict[str, Dict[str, Any]] = {}

        # For each sample in the dataset...
        for s, sample in enumerate(tqdm(
            iterable =  dataset,
            leave =     True,
            unit =      "sample(s)",
            desc =      "Infering samples"
        )):

            # Generate response.
            (response,
             prompt_tokens,
             response_tokens) =             model.generate(prompt = sample.prompt, budget = budget)

            # Extract model's answer.
            extracted:  Optional[str] =     dataset.extract_answer(response = response)
            
            # Grade response.
            correct:    bool =              extracted is not None and       \
                                            extracted.strip().lower() ==    \
                                            sample.ground_truth.strip().lower()

            # Record sample result.
            samples[str(s)] =               {
                                                "prompt":           sample.prompt,
                                                "prompt_tokens":    prompt_tokens,
                                                "ground_truth":     sample.ground_truth,
                                                "response":         response,
                                                "response_tokens":  response_tokens,
                                                "extracted":        extracted,
                                                "correct":          correct
                                            }

        # Ensure output directory exists.
        output_dir.mkdir(parents = True, exist_ok = True)

        # Resolve output file path.
        output_file:    Path =  output_dir / f"{budget_label}-token-budget.json"

        # Compute accuracy.
        num_correct:    int =   sum(1 for s in samples.values() if s["correct"])
        num_samples:    int =   len(samples)
        accuracy:       float = round(num_correct / num_samples, 6) if num_samples > 0 else 0.0

        # Open results file.
        with open(output_file, "w", encoding = "utf-8") as f:

            # Write results.
            dump(
                obj =           {
                                    "model":        model.id,
                                    "dataset":      dataset.id,
                                    "budget":       budget_label,
                                    "num_samples":  dataset.num_samples,
                                    "seed":         seed,
                                    "samples":      samples,
                                },
                fp =            f,
                indent =        2,
                ensure_ascii =  False,
            )