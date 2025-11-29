import time
from typing import List, Optional, Dict, Any, Type
from sentence_transformers import SentenceTransformer
import weaviate.classes.query as wvc

from .base import BaseRAG
from query_weaviate import WeaviateQuery
from models.base import BaseModel
from config import get_settings


class WeaviateRAG(BaseRAG):
    """RAG implementation using Weaviate for retrieval."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedding_model_name: Optional[str] = None,
        weaviate_host: Optional[str] = None,
        weaviate_port: Optional[int] = None,
        weaviate_grpc_port: Optional[int] = None,
        model_class: Optional[Type[BaseModel]] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Weaviate RAG system.

        Args:
            config: Optional configuration dictionary for RAG
            embedding_model_name: Name of the SentenceTransformer model for embeddings (defaults to config)
            weaviate_host: Weaviate server host (defaults to config)
            weaviate_port: Weaviate HTTP port (defaults to config)
            weaviate_grpc_port: Weaviate gRPC port (defaults to config)
            model_class: Optional BaseModel subclass to use for generation
            model_config: Optional configuration for the model (defaults to config)
        """
        super().__init__(config)

        # Load settings
        settings = get_settings()

        # Use provided values or fall back to config
        embedding_model_name = embedding_model_name or settings.embedding_model_name
        weaviate_host = weaviate_host or settings.weaviate_host
        weaviate_port = weaviate_port or settings.weaviate_port
        weaviate_grpc_port = weaviate_grpc_port or settings.weaviate_grpc_port

        # Initialize Weaviate query client
        self.query_client = WeaviateQuery(
            host=weaviate_host,
            port=weaviate_port,
            grpc_port=weaviate_grpc_port,
        )

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"Loaded embedding model: {embedding_model_name}")

        # Store model class and config for lazy initialization
        self.model_class = model_class
        self.model_config = model_config or settings.get_model_config_dict()
        self._model: Optional[BaseModel] = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from Weaviate.

        Args:
            query: Query text to search for
            top_k: Number of documents to retrieve
            filters: Optional filters dict with keys like 'source', 'source_contains', etc.

        Returns:
            List of retrieved documents with properties
        """
        # Convert filter dict to Weaviate Filter object if provided
        weaviate_filter = None
        if filters:
            weaviate_filter = self._build_filter(filters)

        # Retrieve using vector search
        results = self.query_client.query_by_vector(
            query_text=query,
            embedding_model=self.embedding_model,
            limit=top_k,
            filters=weaviate_filter,
            return_metadata=False,
        )

        return results

    def format_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        query: str,
    ) -> str:
        """
        Format retrieved documents into context for the model.

        Args:
            retrieved_docs: List of retrieved documents
            query: Original query

        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant context found."

        context_parts = []
        context_parts.append("Relevant context:")

        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"\n[{i}] {doc.get('question', 'N/A')}")
            if doc.get("context"):
                context_parts.append(f"   Context: {doc.get('context')}")
            if doc.get("answer"):
                context_parts.append(f"   Answer: {doc.get('answer')}")
            if doc.get("source"):
                context_parts.append(f"   Source: {doc.get('source')}")

        return "\n".join(context_parts)

    def set_model(self, model: BaseModel) -> None:
        """
        Set the model instance to use for generation.

        Args:
            model: BaseModel instance
        """
        if not isinstance(model, BaseModel):
            raise TypeError(
                f"Model must be an instance of BaseModel, got {type(model)}"
            )
        self._model = model

    def get_or_create_model(self) -> BaseModel:
        """
        Get existing model or create new one based on model_class and model_config.

        Returns:
            BaseModel instance

        Raises:
            ValueError: If model_class is not set
        """
        if self._model is not None:
            return self._model

        if self.model_class is None:
            raise ValueError(
                "No model set. Either provide model_class in __init__ or call set_model()"
            )

        # Use singleton pattern - get_or_create model instance
        self._model = self.model_class(config=self.model_config)
        return self._model

    def generate_with_context(
        self,
        query: str,
        model: Optional[BaseModel] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG (retrieve + generate).

        Args:
            query: User query/question
            model: Optional BaseModel instance. If None, uses model from config or instance.
            top_k: Number of documents to retrieve
            filters: Optional filters for retrieval
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model parameters

        Returns:
            Dictionary with response, context, and sources
        """
        # Determine which model to use
        if model is None:
            model = self.get_or_create_model()

        if not isinstance(model, BaseModel):
            raise TypeError(
                f"Model must be an instance of BaseModel, got {type(model)}"
            )

        # Ensure model is loaded
        if not model.is_loaded():
            model.load()

        # Step 1: Retrieve relevant context (with timing)
        retrieval_start = time.time()
        retrieved_docs = self.retrieve(query, top_k=top_k, filters=filters)
        retrieval_time = time.time() - retrieval_start
        
        # Extract timing from retrieval if available
        embedding_time = 0
        if retrieved_docs and "_timing" in retrieved_docs[0]:
            timing_info = retrieved_docs[0].pop("_timing")
            embedding_time = timing_info.get("embedding_time", 0)
            retrieval_time = timing_info.get("total_retrieval_time", retrieval_time)

        # Step 2: Format context
        context_str = self.format_context(retrieved_docs, query)
        
        # Calculate context size
        context_chars = len(context_str)
        # Rough token estimate: ~4 characters per token
        context_size = context_chars // 4

        # Step 3: Build prompt with context
        prompt = self._build_rag_prompt(query, context_str)
        prompt_chars = len(prompt)
        prompt_tokens = prompt_chars // 4

        # Step 4: Generate response using the model (with timing)
        generation_start = time.time()
        response = model.generate(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        generation_time = time.time() - generation_start

        # Step 5: Extract sources
        sources = [doc.get("source", "Unknown") for doc in retrieved_docs]

        return {
            "response": response,
            "context": retrieved_docs,
            "sources": sources,
            "num_sources": len(sources),
            "context_size": context_size,
            "context_chars": context_chars,
            "prompt_tokens": prompt_tokens,
            "prompt_chars": prompt_chars,
            "retrieval_time": retrieval_time,
            "embedding_time": embedding_time,
            "generation_time": generation_time,
        }

    def _build_filter(self, filters: Dict[str, Any]) -> Optional[wvc.query.Filter]:
        """
        Build Weaviate Filter from filter dictionary.

        Args:
            filters: Dictionary with filter conditions:
                - 'source': exact source match (single source)
                - 'sources': list of sources for OR matching (multiple sources)
                - 'source_contains': source contains text
                - 'question_contains': question contains text

        Returns:
            Weaviate Filter object or None
        """
        from query_weaviate import (
            filter_by_source,
            filter_by_source_contains,
            filter_by_question_contains,
            filter_by_multiple_sources,
            combine_filters,
        )

        filter_objs = []

        # Handle multiple sources (OR condition) - takes priority over single source
        if "sources" in filters and filters["sources"]:
            sources_list = (
                filters["sources"]
                if isinstance(filters["sources"], list)
                else [filters["sources"]]
            )
            if sources_list:
                filter_objs.append(filter_by_multiple_sources(sources_list))
        elif "source" in filters:
            # Single source match
            filter_objs.append(filter_by_source(filters["source"]))

        if "source_contains" in filters:
            filter_objs.append(filter_by_source_contains(filters["source_contains"]))

        if "question_contains" in filters:
            filter_objs.append(
                filter_by_question_contains(filters["question_contains"])
            )

        if not filter_objs:
            return None

        # Combine multiple filters with AND logic
        if len(filter_objs) == 1:
            return filter_objs[0]
        else:
            # Combine all filters with AND
            combined = filter_objs[0]
            for f in filter_objs[1:]:
                combined = combine_filters(combined, f)
            return combined

    def _build_rag_prompt(self, query: str, context: str) -> str:
        """
        Build a prompt for RAG that includes context.

        Args:
            query: User query
            context: Formatted context from retrieval

        Returns:
            Complete prompt string
        """
        prompt = f"""Use the following context to answer the question. If the context doesn't contain enough information to answer the question, say so.

{context}

Question: {query}

Answer:"""
        return prompt

    def close(self):
        """Close connections and cleanup resources."""
        if hasattr(self, "query_client") and self.query_client:
            self.query_client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
