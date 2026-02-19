"""# parcus.commands.infer.args

Argument definitions & parsing for infer command.
"""

__all__ = ["InferConfig"]

from argparse               import ArgumentParser, _SubParsersAction
from typing                 import override

from parcus.configuration   import CommandConfig
from parcus.registration    import MODEL_REGISTRY

class InferConfig(CommandConfig):
    """# Infer Command Configuration"""

    def __init__(self):
        """# Instantiate Infer Command Configuration."""
        super(InferConfig, self).__init__(
            name =              "infer",
            help =              """Run model inference on a dataset.""",
            subparser_title =   "model-name",
            subparser_help =    """Model who will infer datsaet."""
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
            "--token-budget", "-t",
            dest =      "token_budget",
            type =      int,
            nargs =     "+",
            default =   None,
            help =      """Maximum number of new tokens the model may generate per sample. When not 
                        specified, the model's default generation behavior is used 
                        (unconstrained)."""
        )

        parser.add_argument(
            "--output",
            dest =      "output_path",
            type =      str,
            default =   "output",
            help =      """Path at which inference results will be written. Defaults to 
                        "./output/"."""
        )

        parser.add_argument(
            "--seed", "-s",
            dest =      "seed",
            type =      int,
            default =   1,
            help =      """Random number generation seed. Defaults to 1."""
        )

        # Create sub-parser.
        subparser:  _SubParsersAction = self._create_subparser_(parser = parser)

        # Register models as sub-command.
        MODEL_REGISTRY.register_configurations(subparser = subparser)