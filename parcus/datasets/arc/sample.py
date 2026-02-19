"""# parcus.datasets.arc.sample

ARC Challenge science reasoning sample implementation.
"""

__all__ = ["ARCSample"]

from typing                 import Any, Dict, override

from parcus.datasets.core   import Sample

class ARCSample(Sample):
    """# ARC Challenge Science Reasoning Sample"""

    def __init__(self,
        example:    Dict[str, Any]
    ):
        """# Instantiate ARC-Challenge Sample.

        ## Args:
            * example   (Dict[str, Any]):   Raw dataset example.
        """
        # Initialize sample.
        super(ARCSample, self).__init__(example = example)

    # HELPERS ======================================================================================

    @override
    def _format_prompt_(self) -> str:
        """# Format Raw Example into Model-Ready Prompt.

        ## Returns:
            * str:  Formatted prompt string, ready for tokenization.
        """
        # Format list of choices.
        choices:    str =   "\n".join(
                                f"{label}. {text}"
                                for label, text
                                in zip(
                                    self._raw_["choices"]["label"],
                                    self._raw_["choices"]["text"]
                                )
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
        return self._raw_["answerKey"]