"""# parcus.models.qwen.base

Qwen 2.5 model family.
"""

__all__ = ["Qwen"]

from typing                         import Literal, Optional, Union

from torch                          import device as t_device

from parcus.models.protocol         import Model
from parcus.models.qwen.__args__    import QwenConfig
from parcus.registration            import register_model


@register_model(
    id =        "qwen",
    config =    QwenConfig
)
class Qwen(Model):
    """# Qwen 2.5 Model Family

    ## References:
        * HF:       https://huggingface.co/Qwen
        * Paper:    https://arxiv.org/abs/2407.10671
    """

    def __init__(self,
        parameter_qty:  Literal["0.5B", "1.5B", "3B", "7B", "32B", "72B"],
        max_memory:     Optional[int] =                                     None,
        load_in_4bit:   bool =                                              False,
        offload_path:   str =                                               "offload",
        device:         Union[str, t_device] =                              "auto",
        **kwargs
    ):
        """# Instantiate Qwen Model.

        ## Args:
            * parameter_qty (str):          Model parameter count.
            * max_memory    (int | None):   Limit GPU usage to a certain number of GB.
            * load_in_4bit  (bool):         Load model using 4-bit quantization.
            * offload_path  (str):          Folder for model offloads to share CPU RAM.
            * device        (str | device): Torch device. Defaults to "auto".
        """
        # Initialize model.
        super(Qwen, self).__init__(
            id =            f"qwen-{parameter_qty.lower()}",
            path =          f"Qwen/Qwen2.5-{parameter_qty}-Instruct",
            max_memory =    max_memory,
            load_in_4bit =  load_in_4bit,
            offload_path =  offload_path,
            device =        device
        )
