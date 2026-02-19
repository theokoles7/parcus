"""# parcus.datasets.truthfulqa.dataset

TruthfulQA factual accuracy dataset.
"""

__all__ = ["TruthfulQA"]

from typing                                 import Literal, Optional, override, Type

from parcus.datasets.truthfulqa.__args__    import TruthfulQAConfig
from parcus.datasets.truthfulqa.sample      import TruthfulQASample
from parcus.datasets.core                   import Dataset, Sample
from parcus.registration                    import register_dataset

@register_dataset(
    id =        "truthfulqa",
    config =    TruthfulQAConfig
)
class TruthfulQA(Dataset):
    """# TruthfulQA Factual Accuracy Dataset"""

    def __init__(self,
        subset:         Literal["generation", "multiple_choice"] =  "generation",
        split:          Literal["validation"] =                     "validation",
        num_samples:    Optional[int] =                             None,
        **kwargs
    ):
        """# Instantiate TruthfulQA Dataset.

        ## Args:
            * subset        (str):  Dataset subset being loaded. Defaults to "generation".
            * split         (str):  Dataset split being loaded. Defaults to "validation".
            * num_samples   (int):  Limit number of samples loaded. Defaults to all.
        """
        # Initialize dataset.
        super(TruthfulQA, self).__init__(
            id =            "truthfulqa",
            path =          "truthfulqa/truthful_qa",
            subset =        subset,
            split =         split,
            num_samples =   num_samples
        )

    # PROPERTIES ===================================================================================
    
    @property
    @override
    def _sample_cls_(self) -> Type[Sample]:
        """# Dataset-Specific Sample Type"""
        return TruthfulQASample