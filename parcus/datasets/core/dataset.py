"""# parcus.datasets.core.dataset

Abstract dataset protocol.
"""

__all__ = ["Dataset"]

from abc                            import ABC, abstractmethod
from logging                        import Logger
from re                             import Match, search
from typing                         import Iterator, List, Optional, Type

from datasets                       import Dataset as hf_Dataset, load_dataset

from parcus.datasets.core.sample    import Sample
from parcus.utilities               import get_logger

class Dataset(ABC):
    """# Hugging Face Dataset Wrapper"""

    def __init__(self,
        id:             str,
        path:           str,
        subset:         Optional[str] = None,
        split:          Optional[str] = None,
        num_samples:    Optional[int] = None
    ):
        """# Instantiate Dataset.

        ## Args:
            * id            (str):  Dataset identifier.
            * path          (str):  Hugging Path from which dataset can be loaded.
            * subset        (str):  Dataset subset being loaded. Defaults to None.
            * split         (str):  Dataset split being loaded.
            * num_samples   (int):  Limit number of samples loaded. Defaults to all.
        """
        # Initialize logger.
        self.__logger__:    Logger =    get_logger(f"{id}-dataset")

        # Define properties.
        self._id_:          str =           id
        self._path_:        str =           path
        self._subset_:      Optional[str] = subset
        self._split_:       Optional[str] = split

        # Log initialization.
        self.__logger__.info(f"Loading {path} (subset = {subset}, split = {split})")

        # Load dataset from HuggingFace.
        self._data_:        hf_Dataset =    load_dataset(
                                                path =  self._path_,
                                                name =  self._subset_,
                                                split = self._split_
                                            )
        
        # If a specific number of samples is required...
        if num_samples is not None:

            # Truncate data.
            self._data_ = self._data_.select(indices = range(min(num_samples, len(self._data_))))

            # Log truncation.
            self.__logger__.info(f"Number of samples limited to {self.num_samples}")

    # PROPERTIES ===================================================================================

    @property
    def data(self) -> hf_Dataset:
        """# Underlying HuggingFace Dataset"""
        return self._data_
    
    @property
    def columns(self) -> List[str]:
        """# Dataset Column Names"""
        return self._data_.column_names

    @property
    def id(self) -> str:
        """# Dataset Identifier"""
        return self._id_
    
    @property
    def num_samples(self) -> int:
        """# Number of Samples in Dataset"""
        return len(self._data_)
    
    @property
    def path(self) -> str:
        """# HuggingFace API Path"""
        return self._path_
    
    @property
    @abstractmethod
    def _sample_cls_(self) -> Type[Sample]:
        """# Dataset-Specific Sample Type"""
        pass
    
    @property
    def subset(self) -> str:
        """# Dataset Subset"""
        return self._subset_
    
    # METHODS ======================================================================================

    def extract_answer(self,
        response:   str
    ) -> Optional[str]:
        """# Extract Model's Final Answer from Generated Response.

        ## Args:
            * response  (str):  Raw model response text.

        ## Returns:
            * Optional[str]:    Extracted answer if found.
        """
        # Search for answer in response.
        answer: Match = search(pattern = r"####\s*(.+)", string = response)
        
        # If a match is found, return the extracted answer.
        if answer: return answer.group(1).strip()

        # Otherwise, indicate that no answer was found.
        return None

    # DUNDERS ======================================================================================

    def __getitem__(self,
        key:    int
    ) -> Sample:
        """# Access Dataset Example."""
        return self._sample_cls_(self._data_[key])
    
    def __iter__(self) -> Iterator:
        """# Iterate Over Dataset."""
        return iter(self._sample_cls_(s) for s in self._data_)
    
    def __len__(self) -> int:
        """# Number of Samples in Dataset"""
        return len(self._data_)
    
    def __repr__(self) -> str:
        """# Dataset Object Representation"""
        return  (
                    f"""<{self._id_.upper()}Dataset(path = {self._path_}, """
                    f"""subset = {self._subset_}, split = {self._split_}, """
                    f"""n = {len(self._data_)})>"""
                )