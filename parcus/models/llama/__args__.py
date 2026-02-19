"""# parcus.models.llama.args

Argument definitions & parsing for LLaMA model family.
"""

__all__ = ["LlamaConfig"]

from argparse               import ArgumentParser
from typing                 import override

from parcus.configuration   import ModelConfig


class LlamaConfig(ModelConfig):
    """LLaMA Model Family Configuration & Argument Handler"""

    def __init__(self):
        """# Instantiate LLaMA Model Configuration."""
        super(LlamaConfig, self).__init__(
            name =  "llama",
            help =  "LLaMA model family."
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
            choices =   ["1B", "3B", "8B", "70B"],
            default =   "1B",
            help =      """Model parameter count."""
        )

        # Define generic model arguments.
        super(LlamaConfig, self)._define_arguments_(parser = parser)