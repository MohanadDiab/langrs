"""Detection model implementations."""

try:
    from .grounding_dino import GroundingDINODetector
except (ImportError, AttributeError, ModuleNotFoundError):
    GroundingDINODetector = None

try:
    from .rex_omni import RexOmniDetector
except (ImportError, AttributeError, ModuleNotFoundError):
    RexOmniDetector = None

__all__ = []
if GroundingDINODetector is not None:
    __all__.append("GroundingDINODetector")
if RexOmniDetector is not None:
    __all__.append("RexOmniDetector")
