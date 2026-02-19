"""# parcus.datasets.hellaswag.sample

HellSwag sample implementation.
"""

__all__ = ["HellaSwagSample"]

from typing                 import Any, Dict, override

from parcus.datasets.core   import Sample

# Answer index to letter mapping.
_ANSWER_MAP_ = {0: "A", 1: "B", 2: "C", 3: "D"}

class HellaSwagSample(Sample):
    """# HellaSwag Dataset Sample"""

    def __init__(self,
        example:    Dict[str, Any]
    ):
        """# Instantiate HellaSwag Sample.

        ## Args:
            * example   (Dict[str, Any]):   Raw dataset example.
        """
        # Initialize sample.
        super(HellaSwagSample, self).__init__(example = example)

    # HELPERS ======================================================================================

    @override
    def _format_prompt_(self) -> str:
        """# Format Raw Example into Model-Ready Prompt.

        ## Returns:
            * str:  Formatted prompt string, ready for tokenization.
        """
        # Format list of choices.
        choices:    str =   "\n".join(
                                f"{_ANSWER_MAP_[i]}. {ending}"
                                for i, ending
                                in enumerate(self._raw_["endings"])
                            )

        return  (
                    f"""Choose the most plausible continuation and provide your reasoning. """
                    f"""Present your final answer as a single letter in the format #### ANSWER.\n"""
                    f"""Context: {self._raw_["ctx"]}\n\n{choices}"""
                )

    @override
    def _extract_ground_truth_(self) -> str:
        """# Extract Ground Truth Answer from Raw Example.

        ## Returns:
            * str:  Ground truth answer string.
        """
        return _ANSWER_MAP_[int(self._raw_["label"])]