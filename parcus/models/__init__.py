"""# parcus.models

LLM model factories.
"""

__all__ =   [
                # Protocol
                "Model",

                # Concrete
                "Gemma",
                "Llama",
                "Qwen",
            ]

# Protocol
from parcus.models.protocol import Model
from parcus.models.gemma    import Gemma
from parcus.models.llama    import Llama
from parcus.models.qwen     import Qwen