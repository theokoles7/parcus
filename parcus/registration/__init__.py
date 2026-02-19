"""# parcus.registration

Registry system utilities.
"""

__all__ =   [
                # Registries
                "COMMAND_REGISTRY",
                "DATASET_REGISTRY",
                "MODEL_REGISTRY",

                # Decorators
                "register_command",
                "register_dataset",
                "register_model",
            ]

from parcus.registration.decorators import *
from parcus.registration.registries import *

# Instantiate registries.
COMMAND_REGISTRY:   CommandRegistry =   CommandRegistry()
DATASET_REGISTRY:   DatasetRegistry =   DatasetRegistry()
MODEL_REGISTRY:     ModelRegistry =     ModelRegistry()