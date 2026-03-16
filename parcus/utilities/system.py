"""# parcus.utilities.system

Utility functions for system/hardware information and management.
"""

__all__ =   [
                "determine_device",
                "get_system_core_count",
                "set_hugging_face_token",
                "set_seed",
            ]

from os                 import cpu_count
from pathlib            import Path
from random             import seed as r_seed
from typing             import Literal, Optional, Union

from numpy.random       import seed as np_seed
from torch              import cuda, device as t_device, manual_seed
from torch.backends     import cudnn


def determine_device(
    device: Union[t_device, Literal["auto", "cpu", "cuda"]]
) -> t_device:
    """# Determine Data Processing Device.

    ## Args:
        * device    (str |  device):    Intended device.

    ## Returns:
        * t_device: Best available device based on provided choice.
    """
    # If CPU is chosen, simply return CPU.
    if device == "cpu":     return t_device("cpu")

    # Otherwise, if CUDA is available...
    if cuda.is_available(): return t_device("cuda")

    # If CUDA, is not available, we're using CPU.
    return t_device("cpu")


def get_system_core_count() -> int:
    """# Get System Core Count.

    ## Returns:
        * (int):  Number of available CPU cores.
    """
    # Count cores.
    try:                return cpu_count() or 1

    # Should any complications arise, default to 1.
    except Exception:   return 1


def set_hugging_face_token(
    token_path: Optional[Union[str, Path]] = None
) -> None:
    """# Set Hugging Face API Token.

    ## Args:
        * token_path    (str):  Path from which token (text) file can be loaded.
    """
    from os                 import environ

    from huggingface_hub    import login

    # Resolve path.
    path:   Path =  Path(token_path) if token_path is not None else Path(".hf_token")

    # If path exists...
    if path.exists():
        
        # Set the token.
        environ["HF_TOKEN"] = path.read_text().strip()

        # Log in to Hugging Face Hub.
        # login(token = environ["HF_TOKEN"], add_to_git_credential = True)

    # Otherwise, ask user if they'd like to set one.
    elif input(f"No HuggingFace token found. Set one now? [Y/n] ").lower() not in ["n", "no"]:

        # Prompt user for token and save it.
        token:  str =   input("Enter HuggingFace token: ").strip()

        # Save token to file.
        path.write_text(token)

        # Set token.
        environ["HF_TOKEN"] = token

        # Log in to Hugging Face Hub.
        # login(token = environ["HF_TOKEN"], add_to_git_credential = True)
        

def set_seed(
    seed:   int
) -> None:
    """# Set Random Number Generation Seed.

    ## Args:
        * seed  (int):  Random number generation seed.
    """
    # Set seeds.
    r_seed(seed)
    np_seed(seed)
    manual_seed(seed)

    # If CUDA is available...
    if cuda.is_available():

        # Configure deterministic computing.
        cuda.manual_seed(seed)
        cuda.manual_seed_all(seed)
        cudnn.deterministic =   True
        cudnn.benchmark =       False