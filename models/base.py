from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import hashlib
import json


class BaseModel(ABC):
    """Abstract base class for all model implementations with singleton pattern."""

    _instances: Dict[str, "BaseModel"] = {}

    def __new__(cls, config: Optional[Dict[str, Any]] = None):
        """
        Singleton pattern: return existing instance if one exists for this class and config.

        Args:
            config: Optional configuration dictionary for model-specific settings
        """
        # Create a unique key based on class name and config
        config_dict = config or {}
        config_key = cls._get_config_key(config_dict)
        instance_key = f"{cls.__name__}:{config_key}"

        # Return existing instance if it exists
        if instance_key in cls._instances:
            instance = cls._instances[instance_key]
            # Check if instance is still valid (not unloaded or garbage collected)
            if instance is not None:
                return instance

        # Create new instance
        instance = super().__new__(cls)
        cls._instances[instance_key] = instance
        return instance

    @staticmethod
    def _get_config_key(config: Dict[str, Any]) -> str:
        """
        Generate a hash key from config dictionary for instance identification.

        Args:
            config: Configuration dictionary

        Returns:
            Hash string representing the config
        """
        # Sort config to ensure consistent hashing
        sorted_config = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(sorted_config.encode()).hexdigest()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model.

        Args:
            config: Optional configuration dictionary for model-specific settings
        """
        # Only initialize if not already initialized (singleton pattern)
        if not hasattr(self, "_initialized"):
            self.config = config or {}
            self._model = None
            self._initialized = True

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        pass

    def unload(self) -> None:
        """Unload the model from memory (optional implementation)."""
        # Remove from singleton registry when unloaded
        config_key = self._get_config_key(self.config)
        instance_key = f"{self.__class__.__name__}:{config_key}"
        if instance_key in self._instances:
            del self._instances[instance_key]

    def get_config(self) -> Dict[str, Any]:
        """Get the current model configuration."""
        return self.config.copy()

    @classmethod
    def clear_instance(cls, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Clear a specific instance from the singleton registry.

        Args:
            config: Configuration dictionary. If None, clears all instances of this class.
        """
        if config is None:
            # Clear all instances of this class
            keys_to_remove = [
                key
                for key in cls._instances.keys()
                if key.startswith(f"{cls.__name__}:")
            ]
            for key in keys_to_remove:
                del cls._instances[key]
        else:
            # Clear specific instance
            config_key = cls._get_config_key(config)
            instance_key = f"{cls.__name__}:{config_key}"
            if instance_key in cls._instances:
                del cls._instances[instance_key]

    @classmethod
    def get_instance(
        cls, config: Optional[Dict[str, Any]] = None
    ) -> Optional["BaseModel"]:
        """
        Get existing singleton instance if it exists.

        Args:
            config: Configuration dictionary

        Returns:
            Existing instance or None if not found
        """
        config_dict = config or {}
        config_key = cls._get_config_key(config_dict)
        instance_key = f"{cls.__name__}:{config_key}"
        return cls._instances.get(instance_key, None)
