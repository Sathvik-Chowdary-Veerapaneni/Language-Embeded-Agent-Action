"""
LEAA — Language Grounding Layer

Translates natural-language archery commands into resolved scene targets
and arrow-type selections.
"""

from language_layer.grounding.pipeline import GroundingPipeline

__all__ = ["GroundingPipeline"]
