"""# parcus.datasets.core.sample

Abstract dataset sample protocol.
"""

__all__ = ["Sample"]

from abc        import ABC, abstractmethod
from functools  import cached_property
from typing     import Any, Dict, Optional

class Sample(ABC):
    """# Abstract Dataset Sample"""

    def __init__(self,
        example:    Dict[str, Any]
    ):
        """# Instantiate Sample.

        ## Args:
            * example   (Dict[str, Any]):   Raw dataset example.
        """
        # Store raw example.
        self._raw_: Dict[str, Any] =    example

    # PROPERTIES ===================================================================================

    @cached_property
    def ground_truth(self) -> str:
        """# Ground Truth Answer"""
        return self._extract_ground_truth_()

    @cached_property
    def prompt(self) -> str:
        """# Formatted Model-Ready Prompt"""
        return self._format_prompt_()

    @property
    def raw(self) -> Dict[str, Any]:
        """# Raw Dataset Example"""
        return self._raw_

    # HELPERS ======================================================================================

    @abstractmethod
    def _format_prompt_(self) -> str:
        """# Format Raw Example into Model-Ready Prompt.

        ## Args:
            * example   (Dict[str, Any]):   Raw dataset example.

        ## Returns:
            * str:  Formatted prompt string, ready for tokenization.
        """
        pass

    @abstractmethod
    def _extract_ground_truth_(self) -> str:
        """# Extract Ground Truth Answer from Raw Example.

        ## Args:
            * example   (Dict[str, Any]):   Raw dataset example.

        ## Returns:
            * str:  Ground truth answer string.
        """
        pass
