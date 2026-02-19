"""# parcus.datasets.hellaswag.args

Argument definitions & definitions for HellaSwag dataset.
"""

__all__ = ["HellaSwagConfig"]

from argparse               import _ArgumentGroup, ArgumentParser
from typing                 import override

from parcus.configuration   import DatasetConfig

class HellaSwagConfig(DatasetConfig):
    """HellaSwag Dataset Configuration & Argument Handler"""

    def __init__(self):
        """# Instantiate HellaSwag Dataset Configuration."""
        super(HellaSwagConfig, self).__init__(
            name =  "hellaswag",
            help =  "HellaSwag commonsense reasoning dataset."
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
        # SPLIT ------------------------------------------------------------------------------------
        split:  _ArgumentGroup =    parser.add_argument_group(
                                        title =         "Split",
                                        description =   """Split selection."""
                                    )

        split.add_argument(
            "--split",
            dest =      "split",
            type =      str,
            choices =   ["train", "validation", "test"],
            default =   "validation",
            help =      """Dataset split being loaded. Defaults to "validation"."""
        )

        split.add_argument(
            "--train",
            dest =      "split",
            action =    "store_const",
            const =     "train",
            help =      "Use train split."
        )

        split.add_argument(
            "--validation",
            dest =      "split",
            action =    "store_const",
            const =     "validation",
            help =      "Use validation split."
        )

        split.add_argument(
            "--test",
            dest =      "split",
            action =    "store_const",
            const =     "test",
            help =      "Use test split."
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