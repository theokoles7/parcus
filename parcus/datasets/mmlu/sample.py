"""# parcus.datasets.mmlu.sample

MMLU sample implementation.
"""

__all__ = ["MMLUSample"]

from typing                 import Any, Dict, override

from parcus.datasets.core   import Sample


_ANSWER_MAP_ = {0: "A", 1: "B", 2: "C", 3: "D"}


class MMLUSample(Sample):
    """# MMLU Dataset Sample"""

    def __init__(self,
        example:    Dict[str, Any]
    ):
        """# Instantiate MMLU Sample.

        ## Args:
            * example   (Dict[str, Any]):   Raw dataset example.
        """
        # Initialize sample.
        super(MMLUSample, self).__init__(example = example)

    # HELPERS ======================================================================================

    @override
    def _format_prompt_(self) -> str:
        """# Format Raw Example into Model-Ready Prompt.

        ## Returns:
            * str:  Formatted prompt string, ready for tokenization.
        """
        # Format list of choices.
        choices:    str =   "\n".join(
                                f"{_ANSWER_MAP_[i]}. {choice}"
                                for i, choice
                                in enumerate(self._raw_["choices"])
                            )
        
        return  (
                    f"""Answer the following question and provide your reasoning. """
                    f"""Present your final answer as a single letter in the format #### ANSWER.\n"""
                    f"""Question: {self._raw_["question"]}\n\n{choices}"""
                )

    @override
    def _extract_ground_truth_(self) -> str:
        """# Extract Ground Truth Answer from Raw Example.

        ## Returns:
            * str:  Ground truth answer string.
        """
        return _ANSWER_MAP_[self._raw_["answer"]]