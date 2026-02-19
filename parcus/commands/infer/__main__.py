"""# parcus.commands.infer.main

Main process entry point for infer command.
"""

from typing                         import Any, Dict

from parcus.commands.infer.__args__ import InferConfig
from parcus.registration            import register_command

@register_command(
    id =        "infer",
    config =    InferConfig
)
def infer_entry_point(*args, **kwargs) -> Dict[str, Any]:
    """# Conduct Model Inference on Dataset."""
    print("OK")