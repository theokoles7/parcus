"""# parcus.datasets.mmlu.args

Argument definitions & definitions for MMLU dataset.
"""

__all__ = ["MMLUConfig"]

from argparse               import _ArgumentGroup, ArgumentParser
from typing                 import override

from parcus.configuration   import DatasetConfig

class MMLUConfig(DatasetConfig):
    """MMLU Dataset Configuration & Argument Handler"""

    def __init__(self):
        """# Instantiate MMLU Dataset Configuration."""
        super(MMLUConfig, self).__init__(
            name =  "mmlu",
            help =  "MMLU multi-domain knowledge & question answering dataset."
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Parser Arguments.

        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        # SUBSET -----------------------------------------------------------------------------------
        subset: _ArgumentGroup =    parser.add_argument_group(
                                        title =         "Subset",
                                        description =   """Subset selection."""
                                    )

        subset.add_argument(
            "--subset",
            dest =      "subset",
            type =      str,
            choices =   ["all", "abstract_algebra", "anatomy", "astronomy", "business_ethics",
                         "clinical_knowledge", "college_biology", "college_chemistry",
                         "college_computer_science", "college_mathematics", "college_medicine",
                         "college_physics", "computer_security", "conceptual_physics",
                         "econometrics", "electrical_engineering", "elementary_mathematics",
                         "formal_logic", "global_facts", "high_school_biology",
                         "high_school_chemistry", "high_school_computer_science",
                         "high_school_european_history", "high_school_geography",
                         "high_school_government_and_politics", "high_school_macroeconomics",
                         "high_school_mathematics", "high_school_microeconomics",
                         "high_school_physics", "high_school_psychology",
                         "high_school_statistics", "high_school_us_history",
                         "high_school_world_history", "human_aging", "human_sexuality",
                         "international_law", "jurisprudence", "logical_fallacies",
                         "machine_learning", "management", "marketing", "medical_genetics",
                         "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
                         "philosophy", "prehistory", "professional_accounting",
                         "professional_law", "professional_medicine", "professional_psychology",
                         "public_relations", "security_studies", "sociology",
                         "us_foreign_policy", "virology", "world_religions"],
            default =   "all",
            help =      """Dataset subset being loaded. Defaults to "all"."""
        )

        # SPLIT ------------------------------------------------------------------------------------
        split:  _ArgumentGroup =    parser.add_argument_group(
                                        title =         "Split",
                                        description =   """Split selection."""
                                    )

        split.add_argument(
            "--split",
            dest =      "split",
            type =      str,
            choices =   ["test", "validation", "dev", "auxiliary_train"],
            default =   "test",
            help =      """Dataset split being loaded."""
        )

        split.add_argument(
            "--test",
            dest =      "split",
            action =    "store_const",
            const =     "test",
            help =      "Use test split."
        )

        split.add_argument(
            "--validation",
            dest =      "split",
            action =    "store_const",
            const =     "validation",
            help =      "Use validation split."
        )

        split.add_argument(
            "--dev",
            dest =      "split",
            action =    "store_const",
            const =     "dev",
            help =      "Use dev split."
        )

        # GENERAL ----------------------------------------------------------------------------------
        general:    _ArgumentGroup =    parser.add_argument_group(
                                            title =         "General",
                                            description =   """General dataset configuration."""
                                        )

        general.add_argument(
            "--num-sample", "-n",
            dest =      "num_samples",
            type =      int,
            default =   None,
            help =      """Limit the number of samples loaded from dataset. Defaults to loading 
                        all."""
        )