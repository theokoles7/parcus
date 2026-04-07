"""# parcus.datasets.cruxeval.sample

CRUXEval sample implementation.
"""

__all__ =   [
                "CruxEvalInputSample",
                "CruxEvalOutputSample",
            ]

from typing                 import Any, Dict, override

from parcus.datasets.core   import Sample

class CruxEvalInputSample(Sample):
    """# CRUXEval Dataset Sample for Input Sub-Task"""

    def __init__(self,
        example:    Dict[str, Any]
    ):
        """# Instantiate CRUXEval Input Sample.

        ## Args:
            * example   (Dict[str, Any]):   Raw dataset example.
        """
        # Initialize sample.
        super(CruxEvalInputSample, self).__init__(example = example)

    # HELPERS ======================================================================================

    @override
    def _format_prompt_(self) -> str:
        """# Format Raw Example into Model-Ready Prompt.

        ## Returns:
            * str:  Formatted prompt string, ready for tokenization.
        """
        return  (
                    f"""Given the following Python function, predict the input.\n"""
                    f"""End your answer with #### followed by the exact input as a Python """
                    f"""literal.\nFunction:\n{self._raw_["code"]}\n\nOutput: """
                    f"""{self._raw_["output"]}"""
                )

    @override
    def _extract_ground_truth_(self) -> str:
        """# Extract Ground Truth Answer from Raw Example.

        ## Returns:
            * str:  Ground truth input required.
        """
        return self._raw_["input"]
    

class CruxEvalOutputSample(Sample):
    """# CRUXEval Dataset Sample for Output Sub-Task"""

    def __init__(self,
        example:    Dict[str, Any]
    ):
        """# Instantiate CRUXEval Output Sample.

        ## Args:
            * example   (Dict[str, Any]):   Raw dataset example.
        """
        # Initialize sample.
        super(CruxEvalOutputSample, self).__init__(example = example)

    # HELPERS ======================================================================================

    @override
    def _format_prompt_(self) -> str:
        """# Format Raw Example into Model-Ready Prompt.

        ## Returns:
            * str:  Formatted prompt string, ready for tokenization.
        """
        return  (
                    f"""Given the following Python function, predict the output.\n"""
                    f"""End your answer with #### followed by the exact output as a Python """
                    f"""literal.\nFunction:\n{self._raw_["code"]}\n\nInput: {self._raw_["input"]}"""
                )

    @override
    def _extract_ground_truth_(self) -> str:
        """# Extract Ground Truth Answer from Raw Example.

        ## Returns:
            * str:  Ground truth output required.
        """
        return self._raw_["output"]