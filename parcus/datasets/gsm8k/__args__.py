"""# parcus.datasets.gsm8k.args

Argument definitions & parsing for GSM8K dataset.
"""

__all__ = ["GSM8KConfig"]

from argparse               import _ArgumentGroup, ArgumentParser
from typing                 import override

from parcus.configuration   import DatasetConfig

class GSM8KConfig(DatasetConfig):
    """GSM8K Dataset Configuration & Argument Handler"""

    def __init__(self):
        """# Instantiate GSM8K Dataset Configuration."""
        super(GSM8KConfig, self).__init__(
            name =  "gsm8k",
            help =  "GSM8K grade school math reasoning dataset."
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
            choices =   ["main", "socratic"],
            default =   "main",
            help =      """Dataset subset being loaded. Defaults to "main"."""
        )

        subset.add_argument(
            "--main",
            dest =      "subset",
            action =    "store_const",
            const =     "main",
            help =      """Use main subset."""
        )

        subset.add_argument(
            "--socratic",
            dest =      "subset",
            action =    "store_const",
            const =     "socratic",
            help =      """Use socratic subset."""
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
            choices =   ["train", "test"],
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