"""# parcus.models.protocol

Abstract model protocol.
"""

__all__ = ["Model"]

from abc                import ABC
from logging            import Logger
from typing             import Any, Dict, Optional, Tuple, Union

from torch              import device as t_device, no_grad, Tensor
from transformers       import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, \
                               PreTrainedTokenizerBase

from parcus.utilities   import determine_device, get_logger

class Model(ABC):
    """# Hugging Face Model Wrapper"""

    def __init__(self,
        id:             str,
        path:           str,
        max_memory:     Optional[int] =         None,
        load_in_4bit:   bool =                  False,
        offload_path:   str =                   "offload",
        device:         Union[str, t_device] =  "auto"
    ):
        """# Instantiate Model.

        ## Args:
            * id            (str):              Model identifier.
            * path          (str):              HuggingFace AP path for model loading.
            * max_memory    (int):              Limit GPU usage to a vertain number of GB. Defaults 
                                                to None.
            * load_in_4bit  (bool):             Load model using 4-bit quantization. Defaults to 
                                                False.
            * offload_path  (str):              Folder for model offloads to share CPU RAM. Defaults 
                                                to "offload".
            * device        (str | t_device):   Torch device. Defaults to "auto".
        """
        # Initialize logger.
        self.__logger__:    Logger =            get_logger(f"{id}-model")

        # Define properties.
        self._id_:          str =               id
        self._path_:        str =               path
        self._device_:      t_device =          determine_device(device)
        model_kwargs:       Dict[str, Any] =    {}

        # Log initialization.
        self.__logger__.info(f"Loading {path}")

        # If a maximum memory usage is defined...
        if max_memory is not None:

            # Define memory restriction arguments.
            model_kwargs["max_memory"] =        {0: f"{max_memory}GiB"}
            model_kwargs["device_map"] =        self._device_
            model_kwargs["offload_folder"] =    offload_path

            # Debug memory restriction.
            self.__logger__.info(f"Max memory = {max_memory}GiB; Offload to {offload_path}")

        # If 4-bit quantization is requested...
        if load_in_4bit:

            # Import configuration class.
            from transformers import BitsAndBytesConfig

            # Define configuration.
            model_kwargs["quantization_config"] =   BitsAndBytesConfig(load_in_4bit = True)
            model_kwargs.setdefault("device_map", "auto")

            # Debug configuration.
            self.__logger__.info(f"4-bit quantization enabled")

        # Load model & tokenizer.
        self._model_:       PreTrainedModel =           AutoModelForCausalLM.from_pretrained(
                                                            path,
                                                            **model_kwargs
                                                        )
        self._tokenizer_:   PreTrainedTokenizerBase =   AutoTokenizer.from_pretrained(path)

    # PROPERTIES ===================================================================================

    @property
    def id(self) -> str:
        """# Model Identifier"""
        return self._id_
    
    @property
    def model(self) -> PreTrainedModel:
        """# Underlying HuggingFace Model"""
        return self._model_

    @property
    def path(self) -> str:
        """# HuggingFace API Path"""
        return self._path_
    
    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """# Associated Tokenizer"""
        return self._tokenizer_
    
    # METHODS ======================================================================================

    @no_grad()
    def generate(self,
        prompt: str,
        budget: Optional[int]
    ) -> Tuple[str, int]:
        """# Generate Response from Prompt.

        ## Args:
            * prompt    (str):  Input prompt string.
            * budget    (int):  Token budget. None for unconstrained.

        ## Returns:
            * Tuple[str, int]:
                * str:  Generated response text (excluding prompt).
                * int:  Number of tokens generated.
        """
        # Tokenize prompt.
        inputs:         Dict[str, Tensor] = self._tokenizer_(
                                                text =      prompt,
                                                return_tensors =    "pt"
                                            ).to(self._device_)
        
        # Record prompt length.
        prompt_length:  int =               inputs["input_ids"].shape[-1]

        # Build generation arguments.
        gen_kwargs:     Dict[str, Any] =    {"do_sample": False}

        # If a token budget is defined...
        if budget is not None: gen_kwargs["max_new_tokens"] = budget

        # Generate response.
        output:     Tensor =                self._model_.generate(**inputs, **gen_kwargs)

        # Record response length.
        token_count:    int =               output.shape[-1] - prompt_length
        
        # Decode response (excluding prompt tokens).
        response:       str =               self._tokenizer_.decode(
                                                token_ids =             output[0][prompt_length:],
                                                skip_special_tokens =   True
                                            )
        
        # Provide response & token count.
        return response, token_count