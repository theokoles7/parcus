"""# parcus.datasets.hellaswag.dataset

HellaSwag dataset implementation.

## References:
    * HF:       https://huggingface.co/datasets/Rowan/hellaswag
    * Paper:    https://arxiv.org/abs/1905.07830
"""

__all__ = ["HellaSwag"]

from typing                             import Literal, Optional, override, Type

from parcus.datasets.hellaswag.__args__ import HellaSwagConfig
from parcus.datasets.hellaswag.sample   import HellaSwagSample
from parcus.datasets.core               import Dataset, Sample
from parcus.registration                import register_dataset

@register_dataset(
    id =        "hellaswag",
    config =    HellaSwagConfig
)
class HellaSwag(Dataset):
    """# HellaSwag Commonsense Reasoning Dataset

    ## References:
        * HF:       https://huggingface.co/datasets/Rowan/hellaswag
        * Paper:    https://arxiv.org/abs/1905.07830
    """

    def __init__(self,
        split:          Literal["train", "validation", "test"] =    "validation",
        num_samples:    Optional[int] =                             None,
        **kwargs
    ):
        """# Instantiate HellaSwag Dataset.

        ## Args:
            * split         (str):  Dataset split being loaded. Defaults to "validation".
            * num_samples   (int):  Limit number of samples loaded. Defaults to all.
        """
        # Initialize dataset.
        super(HellaSwag, self).__init__(
            id =            "hellaswag",
            path =          "Rowan/hellaswag",
            subset =        None,
            split =         split,
            num_samples =   num_samples
        )

    # PROPERTIES ===================================================================================
    
    @property
    @override
    def _sample_cls_(self) -> Type[Sample]:
        """# Dataset-Specific Sample Type"""
        return HellaSwagSample