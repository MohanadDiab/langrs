"""Output file path and directory management."""

from pathlib import Path
from datetime import datetime
from typing import Optional


class OutputManager:
    """
    Manages output file paths and directories.
    
    Creates organized output directories with optional timestamping
    and provides convenient path generation.
    """

    def __init__(
        self,
        base_output_path: str,
        create_timestamped: bool = True,
    ):
        """
        Initialize output manager.
        
        Args:
            base_output_path: Base directory for outputs
            create_timestamped: Whether to create timestamped subdirectory
        """
        self.base_path = Path(base_output_path)
        self.create_timestamped = create_timestamped

        if create_timestamped:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = self.base_path / timestamp
        else:
            self.output_path = self.base_path

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

    def get_path(self, filename: str) -> Path:
        """
        Get full path for output file.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Full Path object for the output file
        """
        return self.output_path / filename

    def get_path_str(self, filename: str) -> str:
        """
        Get full path as string for output file.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Full path string for the output file
        """
        return str(self.get_path(filename))

    @property
    def base_dir(self) -> Path:
        """Get base output directory."""
        return self.base_path

    @property
    def output_dir(self) -> Path:
        """Get current output directory."""
        return self.output_path
