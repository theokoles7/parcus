"""# parcus.datasets.arc.args

Argument definitions & parsing for ARC-Challenge dataset.
"""

__all__ = ["ARCConfig"]

from argparse               import _ArgumentGroup, ArgumentParser
from typing                 import override

from parcus.configuration   import DatasetConfig

class ARCConfig(DatasetConfig):
    """ARC Dataset Configuration & Argument Handler"""

    def __init__(self):
        """# Instantiate ARC Dataset Configuration."""
        super(ARCConfig, self).__init__(
            name =  "arc",
            help =  "ARC Challenge science reasoning dataset."
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
            choices =   ["ARC-Challenge", "ARC-Easy"],
            default =   "ARC-Challenge",
            help =      """Dataset subset being loaded. Defaults to "ARC-Challenge"."""
        )

        subset.add_argument(
            "--challenge",
            dest =      "subset",
            action =    "store_const",
            const =     "ARC-Challenge",
            help =      """Use ARC-Challenge subset."""
        )

        subset.add_argument(
            "--easy",
            dest =      "subset",
            action =    "store_const",
            const =     "ARC-Easy",
            help =      """Use ARC-Easy subset."""
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
            choices =   ["train", "validation", "test"],
            default =   "test",
            help =      """Dataset split being loaded."""
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

        # General ----------------------------------------------------------------------------------
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