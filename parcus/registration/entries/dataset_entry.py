"""# parcus.registration.entries.dataset_entry

Dataset registration entry.
"""

__all__ = ["DatasetEntry"]

from typing                     import Type, TYPE_CHECKING

from parcus.registration.core   import Entry

# Defer until runtime.
if TYPE_CHECKING:
    from parcus.configuration   import DatasetConfig
    from parcus.datasets        import Dataset

class DatasetEntry(Entry):
    """# Dataset Registration Entry"""

    def __init__(self,
        id:     str,
        cls:    Type["Dataset"],
        config: Type["DatasetConfig"],
    ):
        """# Instantiate Dataset Registration Entry.

        ## Args:
            * id        (str):                  Dataset identifier.
            * cls       (Type[Dataset]):        Dataset class.
            * config    (Type[DatasetConfig]):  Dataset configuration.
        """
        # Initialize entry.
        super(DatasetEntry, self).__init__(id = id, config = config)

        # Define properties.
        self._cls_: Type["Dataset"] = cls

    # PROPERTIES ===================================================================================

    @property
    def cls(self) -> Type["Dataset"]:
        """# Dataset Class"""
        return self._cls_