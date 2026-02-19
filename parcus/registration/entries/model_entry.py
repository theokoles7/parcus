"""# parcus.registration.entries.model_entry

LLM model registration entry.
"""

__all__ = ["ModelEntry"]

from typing                     import Type, TYPE_CHECKING

from parcus.registration.core   import Entry

# Defer until runtime.
if TYPE_CHECKING:
    from parcus.configuration   import ModelConfig
    from parcus.models          import Model

class ModelEntry(Entry):
    """# Model Registration Entry"""

    def __init__(self,
        id:         str,
        cls:        Type["Model"],
        config:     Type["ModelConfig"]
    ):
        """# Instantiate Model Registration Entry.

        ## Args:
            * id            (str):                  Model identifier.
            * cls           (Type[Model]):          Model class being registered.
            * config        (Type[ModelConfig]):    Model's configuration/argument handler.
        """
        # Initialize entry.
        super(ModelEntry, self).__init__(id = id, config = config)

        # Define properties.
        self._cls_: Type["Model"] = cls

    # PROPERTIES ===================================================================================

    @property
    def cls(self) -> Type["Model"]:
        """# Model Class"""
        return self._cls_