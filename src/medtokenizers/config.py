"""Configuration management for medtokenizers.

This module handles loading configuration from environment variables and .env files.
Provides defaults and validation for the configuration options consumed by the
training entrypoints.

Usage:
    from medtokenizers.config import get_config

    # Get configuration
    config = get_config()
    print(config.wandb_project)
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for medtokenizers.

    Loads from environment variables with sensible defaults.
    """

    # Wandb configuration
    wandb_entity: Optional[str] = None
    wandb_project: str = "medtokenizers"

    # Performance settings
    allow_tf32: bool = True
    channels_last: bool = True

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables.

        Returns:
            Config instance with values from environment

        Example:
            >>> config = Config.from_env()
            >>> print(config.wandb_project)
            medtokenizers
        """
        return cls(
            wandb_entity=os.getenv("WANDB_ENTITY"),
            wandb_project=os.getenv("WANDB_PROJECT", "medtokenizers"),
            allow_tf32=os.getenv("ALLOW_TF32", "true").lower() == "true",
            channels_last=os.getenv("CHANNELS_LAST", "true").lower() == "true",
        )

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not self.wandb_project:
            raise ValueError("wandb_project must be a non-empty string")

        return True


# Global config instance
_config: Optional[Config] = None


def load_dotenv():
    """Load environment variables from .env file if it exists.

    Example:
        >>> load_dotenv()
        # Loads .env file if it exists
    """
    try:
        from dotenv import load_dotenv as _load_dotenv

        env_path = Path(".env")
        if env_path.exists():
            _load_dotenv(env_path)
            return True
        return False
    except ImportError:
        warnings.warn(
            "python-dotenv not installed. Install with: pip install python-dotenv\n"
            "Environment variables from .env file will not be loaded.",
            stacklevel=2,
        )
        return False


def get_config(reload: bool = False) -> Config:
    """Get global configuration instance.

    Args:
        reload: Force reload from environment

    Returns:
        Config instance

    Example:
        >>> config = get_config()
        >>> print(config.wandb_project)
        medtokenizers
    """
    global _config

    if _config is None or reload:
        # Try to load .env file
        load_dotenv()

        # Load from environment
        _config = Config.from_env()

        # Validate
        _config.validate()

    return _config


__all__ = [
    "Config",
    "get_config",
    "load_dotenv",
]
