"""# parcus.models.gemma.args

Argument definitions & parsing for Gemma model family.
"""

__all__ = ["GemmaConfig"]

from argparse               import ArgumentParser
from typing                 import override

from parcus.configuration   import ModelConfig


class GemmaConfig(ModelConfig):
    """Gemma Model Family Configuration & Argument Handler"""

    def __init__(self):
        """# Instantiate Gemma Model Configuration."""
        super(GemmaConfig, self).__init__(
            name =  "gemma",
            help =  "Gemma 3 model family."
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
            choices =   ["1B", "4B", "12B", "27B"],
            default =   "1B",
            help =      """Model parameter count."""
        )

        # Define generic model arguments.
        super(GemmaConfig, self)._define_arguments_(parser = parser)