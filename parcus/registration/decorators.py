"""# parcus.registration.decorators

Function annotation decorators for registration of components.
"""

__all__ =   [
                "register_command",
                "register_dataset",
                "register_model",
            ]

from typing                     import Callable, Type

from parcus.configuration       import CommandConfig, DatasetConfig, ModelConfig


def register_command(
    id:     str,
    config: Type["CommandConfig"]
) -> Callable:
    """# Register Command.

    ## Args:
        * id        (str):                  Command identifier/parser ID.
        * config    (Type[CommandConfig]):  Command's configuration handler class.

    ## Returns:
        * Callable: Registration decorator.
    """
    # Define decorator.
    def decorator(
        entry_point:    Callable
    ) -> Callable:
        """# Command Registration Decorator.

        ## Args:
            * entry_point   (Callable): Command's main process entry point.
        """
        # Load registry.
        from parcus.registration    import COMMAND_REGISTRY

        # Register command.
        COMMAND_REGISTRY.register(
            entry_id =      id,
            config =        config,
            entry_point =   entry_point
        )

        # Expose entry point.
        return entry_point
    
    # Expose decorator.
    return decorator


def register_dataset(
    id:     str,
    config: DatasetConfig
) -> Callable:
    """# Register Dataset.

    ## Args:
        * id        (str):              Dataset identifier.
        * config    (DatasetConfig):    Dataset's configuration/argument handler.

    ## Returns:
        * Callable: Registration decorator.
    """
    # Define decorator.
    def decorator(
        cls:    Type
    ) -> Type:
        """# Dataset Registration Decorator.

        ## Args:
            * cls   (Type): Dataset class being registered.
        """
        # Load registry.
        from parcus.registration    import DATASET_REGISTRY

        # Register dataset.
        DATASET_REGISTRY.register(
            entry_id =      id,
            cls =           cls,
            config =        config,
        )

        # Expose access point.
        return cls
    
    # Expose decorator.
    return decorator


def register_model(
    id:     str,
    config: ModelConfig
) -> Callable:
    """# Register LLM Model.

    ## Args:
        * id        (str):          Model identifier.
        * config    (ModelConfig):  Model's configuration/argument handler.

    ## Returns:
        * Callable: Registration decorator.
    """
    # Define decorator.
    def decorator(
        cls:    Type
    ) -> Callable:
        """# Model Registration Decorator.

        ## Args:
            * cls   (Type): Model class being registered.
        """
        # Load registry.
        from parcus.registration    import MODEL_REGISTRY

        # Register model.
        MODEL_REGISTRY.register(
            entry_id =      id,
            cls =           cls,
            config =        config,
        )

        # Expose access point.
        return cls
    
    # Expose decorator.
    return decorator