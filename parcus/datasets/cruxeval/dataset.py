"""# parcus.datasets.cruxeval.dataset

Code Reasoning, Understanding, and Execution Evaluation dataset.
"""

__all__ = ["CruxEval"]

from ast                                import literal_eval
from re                                 import Match, search
from typing                             import Literal, Optional, override, Type

from parcus.datasets.cruxeval.__args__  import CruxEvalConfig
from parcus.datasets.cruxeval.sample    import *
from parcus.datasets.core               import Dataset, Sample
from parcus.registration                import register_dataset

@register_dataset(
    id =        "cruxeval",
    config =    CruxEvalConfig
)
class CruxEval(Dataset):
    """# Code Reasoning, Understanding, and Execution Evaluation Dataset"""

    def __init__(self,
        subtask:        Literal["input", "output"] =    "input",
        num_samples:    Optional[int] =                 None,
        **kwargs
    ):
        """# Instantiate CRUXEval Dataset.

        ## Args:
            * subtask       (str):  Subtask for which prompts will be formatted. Defaults to 
                                    "input".
            * num_samples   (int):  Limit number of samples loaded. Defaults to all.
        """
        # Initialize dataset.
        super(CruxEval, self).__init__(
            id =            "cruxeval",
            path =          "cruxeval-org/cruxeval",
            subset =        None,
            split =         None,
            num_samples =   num_samples
        )
        
        # Define subtask.
        self._split_:   str =   "test"
        self._subset_:  str =   subtask
        self._subtask_: str =   subtask

    # PROPERTIES ===================================================================================

    @property
    @override
    def _sample_cls_(self) -> Type[Sample]:
        """# Dataset-Specific Sample Type"""
        # Match subtask.
        match self._subtask_:

            # Input prediction.
            case "input":   return CruxEvalInputSample

            # Output prediction.
            case "output":  return CruxEvalOutputSample

    # METHODS ======================================================================================
    
    @override
    def evaluate_answer(self,
        answer:         Optional[str],
        ground_truth:   str
    ) -> bool:
        """# Compare Extracted Answer to Ground Truth.

        ## Args:
            * answer        (str | None):   Answer extracted from model response.
            * ground_truth  (str):          Ground truth answer from dataset.

        ## Returns:
            * bool: True, if answer and ground truth match.
        """
        # If no answer was extracted, no evaluation to be done.
        if answer is None: return False

        try:# Evaluate as Python literals.
            return literal_eval(answer) == literal_eval(ground_truth)
        
        # Default to direct comparison.
        except Exception: return answer.strip() == ground_truth.strip()

    @override
    def extract_answer(self,
        response:   str
    ) -> Optional[str]:
        """# Extract Model's Final Answer from Generated Response.

        ## Args:
            * response  (str):  Raw model response text.

        ## Returns:
            * Optional[str]:    Extracted answer if found.
        """
        # Search for answer in response.
        answer: Match = search(pattern = r"####\s*(.+)", string = response)

        # If a match is found, return the extracted answer.
        if answer: return answer.group(1).strip()

        # Otherwise, indicate that no answer was found.
        return None