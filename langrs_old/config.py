import json
from typing import Dict, Any, Optional

class LangRSConfig:
    """
    Configuration class for LangRS package.
    
    This class handles loading, validating, and accessing configuration settings.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the LangRSConfig object.

        Args:
            config_path (str, optional): Path to a JSON configuration file.
        """
        self.config: Dict[str, Any] = {
            "text_input": "",
            "image_input": "",
            "tile_size": 1000,
            "overlap": 300,
            "tiling": False,
            "evaluation": False,
            "pseudo": False,
            "outlier_methods": ["isolation_forest"],
            "box_threshold": 0.3,
            "text_threshold": 0.75,
            "ground_truth_bb": "",
            "ground_truth_mask": ""
        }
        
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.

        Args:
            config_path (str): Path to the JSON configuration file.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            json.JSONDecodeError: If the configuration file is not valid JSON.
        """
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {config_path}")

    def save_config(self, config_path: str) -> None:
        """
        Save the current configuration to a JSON file.

        Args:
            config_path (str): Path to save the configuration file.
        """
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key (str): The configuration key.
            default (Any, optional): Default value if the key is not found.

        Returns:
            Any: The configuration value.
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key (str): The configuration key.
            value (Any): The value to set.
        """
        self.config[key] = value

    def validate(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        required_keys = ["text_input", "image_input", "tile_size", "overlap", "tiling", "evaluation", "outlier_methods"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration: {key}")

        if not isinstance(self.config["tile_size"], int) or self.config["tile_size"] <= 0:
            raise ValueError("tile_size must be a positive integer")

        if not isinstance(self.config["overlap"], int) or self.config["overlap"] < 0:
            raise ValueError("overlap must be a non-negative integer")

        if not isinstance(self.config["outlier_methods"], list) or len(self.config["outlier_methods"]) == 0:
            raise ValueError("outlier_methods must be a non-empty list")

        if self.config["evaluation"]:
            if not self.config["ground_truth_bb"] or not self.config["ground_truth_mask"]:
                raise ValueError("ground_truth_bb and ground_truth_mask are required when evaluation is True")

    def __str__(self) -> str:
        """
        Return a string representation of the configuration.

        Returns:
            str: A formatted string of the configuration.
        """
        return json.dumps(self.config, indent=4)