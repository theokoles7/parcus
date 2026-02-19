"""# parcus.models.llama.base

LLaMA model family.
"""

__all__ = ["Llama"]

from typing                         import Literal, Optional, Union

from torch                          import device as t_device

from parcus.models.protocol         import Model
from parcus.models.llama.__args__   import LlamaConfig
from parcus.registration            import register_model


# Fixed LLaMA version for each parameter count.
_VERSIONS_ = {
    "1B":   "3.2",
    "3B":   "3.2",
    "8B":   "3.1",
    "70B":  "3.3",
}


@register_model(
    id =        "llama",
    config =    LlamaConfig
)
class Llama(Model):
    """# LLaMA Model Family

    Each parameter count is locked to its corresponding LLaMA release version:
        * 1B, 3B    → LLaMA 3.2
        * 8B        → LLaMA 3.1
        * 70B       → LLaMA 3.3

    ## References:
        * HF:       https://huggingface.co/meta-llama
        * Paper:    https://arxiv.org/abs/2407.21783
    """

    def __init__(self,
        parameter_qty:  Literal["1B", "3B", "8B", "70B"],
        max_memory:     Optional[int] =                     None,
        load_in_4bit:   bool =                              False,
        offload_path:   str =                               "offload",
        device:         Union[str, t_device] =              "auto",
        **kwargs
    ):
        """# Instantiate LLaMA Model.

        ## Args:
            * parameter_qty (str):              Model parameter count.
            * max_memory    (int | None):       Limit GPU usage to a certain number of GB.
            * load_in_4bit  (bool):             Load model using 4-bit quantization.
            * offload_path  (str):              Folder for model offloads to share CPU RAM.
            * device        (str | device):     Torch device. Defaults to "auto".
        """
        # Resolve version from parameter count.
        version:    str =   _VERSIONS_[parameter_qty]

        # Initialize model.
        super(Llama, self).__init__(
            id =            f"llama-{parameter_qty.lower()}",
            path =          f"meta-llama/Llama-{version}-{parameter_qty}-Instruct",
            max_memory =    max_memory,
            load_in_4bit =  load_in_4bit,
            offload_path =  offload_path,
            device =        device
        )