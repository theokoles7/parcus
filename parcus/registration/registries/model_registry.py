"""# parcus.registration.registries.model_registry

LLM model registry system.
"""

__all__ = ["ModelRegistry"]

from typing                         import Dict, override, TYPE_CHECKING

from parcus.registration.core       import Registry
from parcus.registration.entries    import ModelEntry

# Defer until runtime.
if TYPE_CHECKING:
    from parcus.models              import Model

class ModelRegistry(Registry):
    """# LLM Model Registry System"""

    def __init__(self):
        """# Instantiate Model Registry System."""
        # Initialize registry.
        super(ModelRegistry, self).__init__(id = "models")

    # PROPERTIES ===================================================================================

    @override
    @property
    def entries(self) -> Dict[str, ModelEntry]:
        """# Registered Model Entries"""
        return self._entries_.copy()
    
    # METHODS ======================================================================================

    def load_model(self,
        model_id:   str,
        *args,
        **kwargs
    ) -> Model:
        """# Load LLM Model.

        ## Args:
            * model_id  (str):  Identifier of model being loaded.

        ## Returns:
            * Model:    New model instance.
        """
        # Query for registered model.
        entry:  ModelEntry =    self.get_entry(entry_id = model_id)

        # Debug loading.
        self.__logger__.debug(f"Loading {model_id}: {kwargs}")

        # Load model.
        return entry.cls(*args, **kwargs)

    # HELPERS ======================================================================================

    @override
    def _create_entry_(self, **kwargs) -> ModelEntry:
        """# Create Model Entry.

        ## Returns:
            * ModelEntry:   New model entry instance.
        """
        return ModelEntry(**kwargs)

    # DUNDERS ======================================================================================

    @override
    def __getitem__(self,
        model_id:   str
    ) -> ModelEntry:
        """# Query Registered Models.

        ## Args:
            * model_id  (str):  Identifier of model being queried.

        ## Raises:
            * EntryNotFoundError:   If command queried is not registered.

        ## Returns:
            * ModelEntry:   Model entry, if registered.
        """
        return self.get_entry(entry_id = model_id)