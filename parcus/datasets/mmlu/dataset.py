"""# parcus.datasets.mmlu.dataset

MMLU multi-domain knowledge & QA dataset.
"""

__all__ = ["MMLU"]

from typing                         import Optional, override, Type

from parcus.datasets.mmlu.__args__  import MMLUConfig
from parcus.datasets.mmlu.sample    import MMLUSample
from parcus.datasets.core           import Dataset, Sample
from parcus.registration            import register_dataset

@register_dataset(
    id =        "mmlu",
    config =    MMLUConfig
)
class MMLU(Dataset):
    """# MMLU Multi-Domain Knowledge & QA Dataset"""

    def __init__(self,
        subset:         str =           "all",
        split:          str =           "test",
        num_samples:    Optional[int] = None,
        **kwargs
    ):
        """# Instantiate MMLU Dataset.

        ## Args:
            * subset        (str):  Dataset subset being loaded. Defaults to "all".
            * split         (str):  Dataset split being loaded.
            * num_samples   (int):  Limit number of samples loaded. Defaults to all.
        """
        # Initialize dataset.
        super(MMLU, self).__init__(
            id =            "mmlu",
            path =          "cais/mmlu",
            subset =        subset,
            split =         split,
            num_samples =   num_samples
        )

    # PROPERTIES ===================================================================================
    
    @property
    @override
    def _sample_cls_(self) -> Type[Sample]:
        """# Dataset-Specific Sample Type"""
        return MMLUSample