"""# parcus.args

Argument definitions & parsing for parcus application.
"""

__all__ = ["parse_parcus_arguments"]

from argparse               import _ArgumentGroup, ArgumentParser, Namespace, _SubParsersAction
from typing                 import Optional, Sequence

from parcus.registration    import COMMAND_REGISTRY

def parse_parcus_arguments(
    args:       Optional[Sequence[str]] =   None,
    namespace:  Optional[Namespace] =       None
) -> Namespace:
    """# Parse parcus Arguments.

    ## Args:
        * args      (Sequence[str] | None): Sequence of string arguments.
        * namespace (Namespace | None):     Mapping of arguments to their values.

    ## Returns:
        * Namespace:    Mapping of arugments & their values.
    """
    # Initilize parser.
    parser:     ArgumentParser =    ArgumentParser(
                                        prog =          "parcus",
                                        description =   """Experiments in analyzing the correlation 
                                                        and effects of token budgets on a language 
                                                        model's ability to reason and generate 
                                                        accurate responses."""
                                    )
    
    # Initialize sub-parser.
    subparser:  _SubParsersAction = parser.add_subparsers(
                                        title =         "parcus-command",
                                        dest =          "parcus_command",
                                        help =          """Parcus command being executed.""",
                                        description =   """Parcus command being executed."""
                                    )
    
    # +============================================================================================+
    # | BEGIN ARGUMENTS                                                                            |
    # +============================================================================================+

    # LOGGING ======================================================================================
    logging:     _ArgumentGroup =    parser.add_argument_group(
                                        title =         "Logging",
                                        description =   """Logging utility configuration."""
                                    )

    logging.add_argument(
        "--logging-level",
        dest =      "logging_level",
        type =      str,
        choices =   ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"],
        default =   "INFO",
        help =      """Minimum logging level (DEBUG < INFO < WARNING < ERROR < CRITICAL). 
                    Defaults to "INFO"."""
    )

    logging.add_argument(
        "--logging-path",
        dest =      "logging_path",
        type =      str,
        default =   "logs",
        help =      """Path at which logs will be written. Defaults to "./logs/"."""
    )

    logging.add_argument(
        "--debug", "-v",
        dest =      "logging_level",
        action =    "store_const",
        const =     "DEBUG",
        help =      """Set logging level to DEBUG."""
    )
    
    # +============================================================================================+
    # | END ARGUMENTS                                                                              |
    # +============================================================================================+

    # Register commands.
    COMMAND_REGISTRY.register_configurations(subparser = subparser)

    # Parse arguments.
    return parser.parse_args(args, namespace)