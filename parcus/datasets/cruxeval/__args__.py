"""# parcus.datasets.cruxeval.args

Argument definitions & parsing for CRUXEval dataset.
"""

__all__ = ["CruxEvalConfig"]

from argparse               import _ArgumentGroup, ArgumentParser
from typing                 import override

from parcus.configuration   import DatasetConfig

class CruxEvalConfig(DatasetConfig):
    """# CRUXEval Dataset Configuration & Argument Handler"""

    def __init__(self):
        """# Instantiate CRUXEval Dataset Configuration."""
        super(CruxEvalConfig, self).__init__(
            name =  "cruxeval",
            help =  "Code Reasoning, Understanding, and Execution Evaluation"
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
        # TASK -------------------------------------------------------------------------------------
        subtask:    _ArgumentGroup =    parser.add_argument_group(
                                            title =         "Sub-Task",
                                            description =   """Sub-task selection."""
                                        )
        
        subtask.add_argument(
            "--subtask",
            dest =      "subtask",
            type =      str,
            choices =   ["input", "output"],
            default =   "input",
            help =      """Task for which prompts will be formatted. Defaults to "input"."""
        )

        subtask.add_argument(
            "--input",
            dest =      "subtask",
            action =    "store_const",
            const =     "input",
            help =      """Given a funtion and output, predict the input."""
        )

        subtask.add_argument(
            "--output",
            dest =      "subtask",
            action =    "store_const",
            const =     "output",
            help =      """Given a function and input, predict the output."""
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