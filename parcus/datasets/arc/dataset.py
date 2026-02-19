"""# parcus.datasets.arc.dataset

ARC Challenge science reasoning dataset.
"""

__all__ = ["ARC"]

from typing                         import Literal, Optional, override, Type

from parcus.datasets.arc.__args__   import ARCConfig
from parcus.datasets.arc.sample     import ARCSample
from parcus.datasets.core           import Dataset, Sample
from parcus.registration            import register_dataset

@register_dataset(
    id =        "arc",
    config =    ARCConfig
)
class ARC(Dataset):
    """# ARC Challenge Science Reasoning Dataset"""

    def __init__(self,
        subset:         Literal["ARC-Challenge", "ARC-Easy"] =      "ARC-Challenge",
        split:          Literal["train", "validation", "test"] =    "test",
        num_samples:    Optional[int] =                             None,
        **kwargs
    ):
        """# Instantiate ARC Dataset.

        ## Args:
            * subset        (str):  Dataset subset being loaded. Defaults to None.
            * split         (str):  Dataset split being loaded.
            * num_samples   (int):  Limit number of samples loaded. Defaults to all.
        """
        # Initialize dataset.
        super(ARC, self).__init__(
            id =            "arc",
            path =          "allenai/ai2_arc",
            subset =        subset,
            split =         split,
            num_samples =   num_samples
        )

    # PROPERTIES ===================================================================================
    
    @property
    @override
    def _sample_cls_(self) -> Type[Sample]:
        """# Dataset-Specific Sample Type"""
        return ARCSample