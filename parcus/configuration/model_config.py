"""# parcus.configuration.model_config

LLM model configuration & parsing handler.
"""

__all__ = ["ModelConfig"]

from argparse                       import ArgumentParser, _SubParsersAction
from typing                         import override

from parcus.configuration.protocol  import Config

class ModelConfig(Config):
    """# LLM Model Configuration & Argument Handler"""

    def __init__(self,
        name:   str,
        help:   str
    ):
        """# Instantiate Model Configuration.

        ## Args:
            * name  (str):  Model identifier.
            * help  (str):  Brief model description.
        """
        # Initialize configuration.
        super(ModelConfig, self).__init__(
            parser_id =         name,
            parser_help =       help,
            subparser_title =   "dataset-id",
            subparser_help =    """Dataset with which model will execute action."""
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
        from parcus.registration            import DATASET_REGISTRY
        
        parser.add_argument(
            "--max-memory",
            dest =      "max_memory",
            type =      int,
            help =      """Limit GPU usage to a certain number of GB."""
        )

        parser.add_argument(
            "--4-bit",
            dest =      "load_in_4bit",
            action =    "store_true",
            default =   False,
            help =      """Load model using 4-bit quantization."""
        )

        parser.add_argument(
            "--offload-path",
            dest =      "offload_path",
            type =      str,
            default =   "offload",
            help =      """Folder at which model offloads can be stored to share CPU RAM. Defaults 
                        to "./offload/"."""
        )

        # Create subparser.
        subparser:  _SubParsersAction = self._create_subparser_(parser = parser)

        # Register datasets as sub-command.
        DATASET_REGISTRY.register_configurations(subparser = subparser)