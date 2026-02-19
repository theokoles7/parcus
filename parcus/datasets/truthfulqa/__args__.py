"""# parcus.datasets.truthfulqa.args

Argument definitions & definitions for TruthfulQA dataset.

## References:
    * HF:       https://huggingface.co/datasets/truthfulqa/truthful_qa
    * Paper:    https://arxiv.org/abs/2109.07958
"""

__all__ = ["TruthfulQAConfig"]

from argparse               import _ArgumentGroup, ArgumentParser
from typing                 import override

from parcus.configuration   import DatasetConfig

class TruthfulQAConfig(DatasetConfig):
    """TruthfulQA Dataset Configuration & Argument Handler

    ## References:
        * HF:       https://huggingface.co/datasets/truthfulqa/truthful_qa
        * Paper:    https://arxiv.org/abs/2109.07958
    """

    def __init__(self):
        """# Instantiate TruthfulQA Dataset Configuration."""
        super(TruthfulQAConfig, self).__init__(
            name =  "truthfulqa",
            help =  "TruthfulQA factual accuracy dataset."
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Parser Arguments.

        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        # SUBSET -----------------------------------------------------------------------------------
        subset: _ArgumentGroup =    parser.add_argument_group(
                                        title =         "Subset",
                                        description =   """Subset selection."""
                                    )

        subset.add_argument(
            "--subset",
            dest =      "subset",
            type =      str,
            choices =   ["generation", "multiple_choice"],
            default =   "generation",
            help =      """Dataset subset being loaded. Defaults to "generation"."""
        )

        # SPLIT ------------------------------------------------------------------------------------
        split:  _ArgumentGroup =    parser.add_argument_group(
                                        title =         "Split",
                                        description =   """Split selection."""
                                    )

        split.add_argument(
            "--split",
            dest =      "split",
            type =      str,
            choices =   ["validation"],
            default =   "validation",
            help =      """Dataset split being loaded. Only "validation" is available."""
        )

        # GENERAL ----------------------------------------------------------------------------------
        general:    _ArgumentGroup =    parser.add_argument_group(
                                            title =         "General",
                                            description =   """General dataset configuration."""
                                        )

        general.add_argument(
            "--num-sample", "-n",
            dest =      "num_samples",
            type =      int,
            default =   None,
            help =      """Limit the number of samples loaded from dataset. Defaults to loading 
                        all."""
        )