"""# parcus.configuration.dataset_config

Dataset configuration & parsing handler.
"""

__all__ = ["DatasetConfig"]

from argparse                       import ArgumentParser
from typing                         import override

from parcus.configuration.protocol  import Config

class DatasetConfig(Config):
    """# Dataset Configuration & Argument Handler"""

    def __init__(self,
        name:   str,
        help:   str
    ):
        """# Instantiate Dataset Configuration.

        ## Args:
            * name  (str):  Dataset identifier.
            * help  (str):  Brief dataset description.
        """
        # Initialize configuration.
        super(DatasetConfig, self).__init__(
            parser_id =     name,
            parser_help =   help
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
            "--num-sample", "-n",
            dest =      "num_samples",
            type =      int,
            default =   None,
            help =      """Limit the number of samples loaded from dataset. Defaults to loading 
                        all."""
        )