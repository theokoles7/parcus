"""# parcus.datasets.gsm8k.sample

GSM8K sample implementation.
"""

__all__ = ["GSM8KSample"]

from typing                 import Any, Dict, override

from parcus.datasets.core   import Sample

class GSM8KSample(Sample):
    """# GSM8K Dataset Sample"""
    
    def __init__(self,
        example:    Dict[str, Any]
    ):
        """# Instantiate GSM8K Sample.

        ## Args:
            * example   (Dict[str, Any]):   Raw dataset example.
        """
        # Initialize sample.
        super(GSM8KSample, self).__init__(example = example)

    # HELPERS ======================================================================================

    @override
    def _format_prompt_(self) -> str:
        """# Format Raw Example into Model-Ready Prompt.

        ## Returns:
            * str:  Formatted prompt string, ready for tokenization.
        """
        return  (
                    f"""Solve the following math problem and show your work. """
                    f"""Present your final numeric answer in the format #### ANSWER.\n"""
                    f"""Question: {self._raw_["question"]}"""
                )

    @override
    def _extract_ground_truth_(self) -> str:
        """# Extract Ground Truth Answer from Raw Example.

        Parses the final numeric answer from the native `#### <number>` pattern.

        ## Returns:
            * str:  Ground truth answer string.
        """
        return self._raw_["answer"].split("####")[-1].strip()