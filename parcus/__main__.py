"""# parcus.main

Primary application process.
"""

__all__ = ["parcus_entry_point"]

from argparse   import Namespace
from logging    import Logger
from typing     import Any

def parcus_entry_point(*args, **kwargs) -> Any:
    """# Execute Parcus Command.

    ## Returns:
        * Any:  Data returned from sub-process(es).
    """
    # Package imports.
    from parcus.__args__       import parse_parcus_arguments
    from parcus.registration   import COMMAND_REGISTRY
    from parcus.utilities      import configure_logger

    # Parse arguments.
    arguments:  Namespace = parse_parcus_arguments(*args, **kwargs)

    # Initialize logger.
    logger:     Logger =    configure_logger(
                                logging_level = arguments.logging_level,
                                logging_path =  arguments.logging_path
                            )
    
    # Debug arguments.
    logger.debug(f"Parcus arguments: {vars(arguments)}")

    try:# Dispatch to command.
        COMMAND_REGISTRY.dispatch(command_id = arguments.parcus_command, **vars(arguments))

    # Catch wildcard errors.
    except Exception as e:  logger.critical(f"Unexpected error: {e}", exc_info = True)

    # Exit gracefully.
    finally:                logger.debug("Exiting...")


if __name__ == "__main__": parcus_entry_point()