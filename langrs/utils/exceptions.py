"""Custom exception hierarchy for LangRS."""


class LangRSError(Exception):
    """Base exception for all LangRS errors."""

    def __init__(self, message: str, *args, **kwargs):
        self.message = message
        super().__init__(message, *args, **kwargs)


class ModelLoadError(LangRSError):
    """Error loading model weights or initializing model."""

    pass


class ImageLoadError(LangRSError):
    """Error loading or processing image."""

    pass


class DetectionError(LangRSError):
    """Error during object detection."""

    pass


class SegmentationError(LangRSError):
    """Error during segmentation."""

    pass


class ConfigurationError(LangRSError):
    """Error in configuration."""

    pass
