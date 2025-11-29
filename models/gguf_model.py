from typing import Optional, Dict, Any
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from .base import BaseModel


class GGUFModel(BaseModel):
    """GGUF model implementation using llama-cpp-python."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GGUF model.

        Config options:
            - model_repo_id: Hugging Face repo ID (default: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
            - model_filename: Model filename (default: "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
            - n_ctx: Context window size (default: 4096)
            - n_threads: Number of threads (default: None, auto-detect)
            - n_gpu_layers: GPU layers for Metal acceleration (default: 1)
        """
        super().__init__(config)
        self.model_repo_id = self.config.get(
            "model_repo_id", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        )
        self.model_filename = self.config.get(
            "model_filename", "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        )
        self.n_ctx = self.config.get("n_ctx", 4096)
        self.n_threads = self.config.get("n_threads", None)
        self.n_gpu_layers = self.config.get("n_gpu_layers", 1)
        self._model = None

    def load(self) -> None:
        """Load the GGUF model from Hugging Face."""
        if self.is_loaded():
            print("Model already loaded")
            return

        print(f"Loading GGUF model: {self.model_repo_id}/{self.model_filename}")
        print("This will download the model on first run (~4.5GB)")

        # Download the model file from Hugging Face
        model_path = hf_hub_download(
            repo_id=self.model_repo_id,
            filename=self.model_filename,
            local_dir=None,  # Use Hugging Face cache
        )

        print(f"Model downloaded to: {model_path}")

        # Load the GGUF model with llama.cpp
        self._model = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

        print("Model loaded successfully!")
        print("Model uses approximately 4-5GB of RAM (4-bit quantization)")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stop: Optional[list] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response using the GGUF model.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            repeat_penalty: Penalty for repeating tokens
            stop: List of stop sequences
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        # Format prompt for Mistral Instruct
        # llama.cpp handles the <s> token automatically
        formatted_prompt = f"[INST] {prompt} [/INST]"

        # Default stop sequences
        if stop is None:
            stop = ["</s>", "[INST]", "[/INST]"]

        # Generate response using llama.cpp
        # Optimization: Use top_k for faster sampling if not provided
        generation_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stop": stop,
            "echo": False,  # Don't echo the prompt
        }
        
        # Add top_k for faster sampling (if not in kwargs)
        if "top_k" not in kwargs:
            generation_kwargs["top_k"] = 40  # Limit candidate tokens for speed
        
        # Merge with any additional kwargs
        generation_kwargs.update(kwargs)
        
        output = self._model(formatted_prompt, **generation_kwargs)

        # Extract the generated text
        response = output["choices"][0]["text"].strip()
        return response

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            # llama.cpp doesn't have an explicit unload, but we can clear the reference
            del self._model
            self._model = None
            print("Model unloaded from memory")

        # Call parent unload to remove from singleton registry
        super().unload()
