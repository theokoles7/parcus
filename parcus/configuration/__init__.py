"""# parcus.configuration

Configuration protocols.
"""

__all__ =   [
                # Protocol
                "Config",

                # Concrete
                "CommandConfig",
                "DatasetConfig",
                "ModelConfig",
            ]

from parcus.configuration.command_config    import CommandConfig
from parcus.configuration.dataset_config    import DatasetConfig
from parcus.configuration.model_config      import ModelConfig
from parcus.configuration.protocol          import Config