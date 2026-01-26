"""Detection model implementations."""

from .grounding_dino import GroundingDINODetector

try:
    from .rex_omni import RexOmniDetector
except ImportError:
    RexOmniDetector = None

__all__ = ["GroundingDINODetector"]
if RexOmniDetector is not None:
    __all__.append("RexOmniDetector")
