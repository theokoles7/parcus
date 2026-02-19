"""# parcus.datasets.gsm8k.dataset

GSM8K math reasoning dataset.
"""

__all__ = ["GSM8K"]

from typing                         import Literal, Optional, override, Type

from parcus.datasets.gsm8k.__args__ import GSM8KConfig
from parcus.datasets.gsm8k.sample   import GSM8KSample
from parcus.datasets.core           import Dataset, Sample
from parcus.registration            import register_dataset

@register_dataset(
    id =        "gsm8k",
    config =    GSM8KConfig
)
class GSM8K(Dataset):
    """# GSM8K Math Reasoning Dataset"""

    def __init__(self,
        subset:         Literal["main", "socratic"] =   "main",
        split:          Literal["train", "test"] =      "test",
        num_samples:    Optional[int] =                 None,
        **kwargs
    ):
        """# Instantiate GSM8K Dataset.

        ## Args:
            * subset        (str):  Dataset subset being loaded. Defaults to "main".
            * split         (str):  Dataset split being loaded.
            * num_samples   (int):  Limit number of samples loaded. Defaults to all.
        """
        # Initialize dataset.
        super(GSM8K, self).__init__(
            id =            "gsm8k",
            path =          "openai/gsm8k",
            subset =        subset,
            split =         split,
            num_samples =   num_samples
        )

    # PROPERTIES ===================================================================================
    
    @property
    @override
    def _sample_cls_(self) -> Type[Sample]:
        """# Dataset-Specific Sample Type"""
        return GSM8KSample