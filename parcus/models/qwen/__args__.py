"""# parcus.models.qwen.args

Argument definitions & parsing for QWEN family models.
"""

from argparse               import ArgumentParser
from typing                 import override

from parcus.configuration   import ModelConfig

class QwenConfig(ModelConfig):
    """# Qwen Model Family Configuration & Argument Handler"""

    def __init__(self):
        """# Instantiate Qwen Model Configuration."""
        super(QwenConfig, self).__init__(
            name =  "qwen",
            help =  "Qwen 2.5 model family."
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
        parser.add_argument(
            "--parameters", "-p",
            dest =      "parameter_qty",
            type =      str,
            choices =   ["0.5B", "1.5B", "3B", "7B", "32B", "72B"],
            default =   "0.5B",
            help =      """Model parameter count."""
        )

        # Define generic model arguments.
        super(QwenConfig, self)._define_arguments_(parser = parser)