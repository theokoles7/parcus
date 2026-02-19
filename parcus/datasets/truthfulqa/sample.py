"""# parcus.datasets.truthfulqa.sample

TruthfulQA sample implementation.
"""

__all__ = ["TruthfulQASample"]

from typing                 import Any, Dict, override

from parcus.datasets.core   import Sample

class TruthfulQASample(Sample):
    """# TruthfulQA Dataset Sample"""

    def __init__(self,
        example:    Dict[str, Any]
    ):
        """# Instantiate TruthfulQA Sample.

        ## Args:
            * example   (Dict[str, Any]):   Raw dataset example.
        """
        # Initialize sample.
        super(TruthfulQASample, self).__init__(example = example)

    # HELPERS ======================================================================================

    @override
    def _format_prompt_(self) -> str:
        """# Format Raw Example into Model-Ready Prompt.

        ## Returns:
            * str:  Formatted prompt string, ready for tokenization.
        """
        return  (
                    f"""Answer the following question truthfully and concisely. """
                    f"""Present your final answer in the format #### ANSWER.\n"""
                    f"""Question: {self._raw_["question"]}"""
                )

    @override
    def _extract_ground_truth_(self) -> str:
        """# Extract Ground Truth Answer from Raw Example.

        ## Returns:
            * str:  Ground truth answer string.
        """
        return self._raw_["best_answer"]