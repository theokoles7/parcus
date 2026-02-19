"""# parcus.models.gemma.base

Gemma 3 model family.
"""

__all__ = ["Gemma"]

from typing                             import Literal, Optional, Union

from torch                              import device as t_device

from parcus.models.protocol             import Model
from parcus.models.gemma.__args__       import GemmaConfig
from parcus.registration                import register_model


@register_model(
    id =        "gemma",
    config =    GemmaConfig
)
class Gemma(Model):
    """# Gemma 3 Model Family

    ## References:
        * HF:       https://huggingface.co/google
        * Paper:    https://arxiv.org/abs/2503.19786
    """

    def __init__(self,
        parameter_qty:  Literal["1B", "4B", "12B", "27B"],
        max_memory:     Optional[int] =             None,
        load_in_4bit:   bool =                      False,
        offload_path:   str =                       "offload",
        device:         Union[str, t_device] =      "auto",
        **kwargs
    ):
        """# Instantiate Gemma Model.

        ## Args:
            * parameter_qty (str):          Model parameter count.
            * max_memory    (int | None):   Limit GPU usage to a certain number of GB.
            * load_in_4bit  (bool):         Load model using 4-bit quantization.
            * offload_path  (str):          Folder for model offloads to share CPU RAM.
            * device        (str | device): Torch device. Defaults to "auto".
        """
        # Initialize model.
        super(Gemma, self).__init__(
            id =            f"gemma-{parameter_qty.lower()}",
            path =          f"google/gemma-3-{parameter_qty.lower()}-it",
            max_memory =    max_memory,
            load_in_4bit =  load_in_4bit,
            offload_path =  offload_path,
            device =        device
        )