from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from models.base import BaseModel


class BaseRAG(ABC):
    """Abstract base class for Retrieval-Augmented Generation implementations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG system.
        
        Args:
            config: Optional configuration dictionary for RAG-specific settings
        """
        self.config = config or {}

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from the knowledge base.
        
        Args:
            query: Query text to search for
            top_k: Number of documents to retrieve
            filters: Optional filters to apply to the search
            
        Returns:
            List of retrieved documents with their properties (e.g., question, answer, context, source)
        """
        pass

    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        model: "BaseModel",
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG (retrieve context + generate answer).
        
        Args:
            query: User query/question
            model: Language model instance implementing BaseModel
            top_k: Number of documents to retrieve
            filters: Optional filters to apply to retrieval
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing:
                - response: Generated answer text
                - context: Retrieved context documents
                - sources: List of source documents used
        """
        pass

    @abstractmethod
    def format_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        query: str,
    ) -> str:
        """
        Format retrieved documents into a context string for the model.
        
        Args:
            retrieved_docs: List of retrieved documents
            query: Original query for context
            
        Returns:
            Formatted context string to include in the prompt
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get the current RAG configuration."""
        return self.config.copy()

