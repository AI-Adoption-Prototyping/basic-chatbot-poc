"""
Configuration management using Pydantic BaseSettings.
Loads configuration from environment variables and .env files.
"""

from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    # Environment
    environment: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # Model Configuration
    model_type: str = Field(default="gguf", description="Type of model to use")
    model_repo_id: str = Field(
        default="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        description="Hugging Face repository ID for GGUF models",
    )
    model_filename: str = Field(
        default="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        description="Model filename for GGUF models",
    )
    model_n_ctx: int = Field(default=4096, description="Context window size")
    model_n_threads: Optional[int] = Field(
        default=None, description="Number of threads (None = auto-detect)"
    )

    @field_validator("model_n_threads", mode="before")
    @classmethod
    def parse_model_n_threads(cls, v):
        """Convert empty string from .env to None."""
        if v == "" or v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        if isinstance(v, str):
            # Handle string "None" or empty string
            if v.strip().lower() in ("none", "null", ""):
                return None
            return int(v)
        return v

    model_n_gpu_layers: int = Field(
        default=1, description="GPU layers for Metal acceleration (M1/M2 Mac)"
    )

    # Weaviate Configuration
    weaviate_host: str = Field(default="localhost", description="Weaviate server host")
    weaviate_port: int = Field(default=8080, description="Weaviate HTTP port")
    weaviate_grpc_port: int = Field(default=50051, description="Weaviate gRPC port")
    weaviate_collection_name: str = Field(
        default="RAGEntry", description="Weaviate collection name for RAG entries"
    )

    # Embedding Model Configuration
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name for embeddings",
    )

    # Hugging Face API (Security-sensitive)
    huggingface_apikey: Optional[str] = Field(
        default=None, description="Hugging Face API key (required for some modules)"
    )

    # Database Configuration
    db_path: str = Field(
        default="chat_history.db", description="SQLite database file path"
    )

    # FastAPI Configuration
    fastapi_host: str = Field(default="0.0.0.0", description="FastAPI server host")
    fastapi_port: int = Field(default=8888, description="FastAPI server port")
    fastapi_reload: bool = Field(
        default=False, description="Enable auto-reload for development"
    )

    # RAG Default Configuration
    rag_top_k: int = Field(
        default=5, description="Default number of documents to retrieve"
    )
    rag_max_tokens: int = Field(
        default=200, description="Default maximum tokens to generate"
    )
    rag_temperature: float = Field(
        default=0.7, description="Default sampling temperature"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def get_model_config_dict(self) -> dict:
        """Get model configuration as dictionary for model initialization."""
        config = {}

        if self.model_type == "gguf":
            config["model_repo_id"] = self.model_repo_id
            config["model_filename"] = self.model_filename
            config["n_ctx"] = self.model_n_ctx
            if self.model_n_threads is not None:
                config["n_threads"] = self.model_n_threads
            config["n_gpu_layers"] = self.model_n_gpu_layers

        return config

    def get_weaviate_config(self) -> dict:
        """Get Weaviate configuration as dictionary."""
        return {
            "host": self.weaviate_host,
            "port": self.weaviate_port,
            "grpc_port": self.weaviate_grpc_port,
            "collection_name": self.weaviate_collection_name,
        }

    def get_rag_config(self) -> dict:
        """Get RAG default configuration as dictionary."""
        return {
            "top_k": self.rag_top_k,
            "max_tokens": self.rag_max_tokens,
            "temperature": self.rag_temperature,
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global _settings
    _settings = None
