#!/usr/bin/env python3
"""
Example usage of the RAG system with Weaviate.
"""
from rag import WeaviateRAG
from models import GGUFModel
from config import get_settings


def main():
    # Load settings from environment/config
    settings = get_settings()

    # Initialize RAG system with configuration from settings
    print("Initializing RAG system...")
    print(f"Using configuration from environment/.env")

    rag = WeaviateRAG(
        # All config values default to settings, but can be overridden
        model_class=GGUFModel,
        model_config=None,  # Will use settings.get_model_config_dict()
    )

    # Initialize language model (will use singleton pattern)
    print("Loading language model...")
    model = rag.get_or_create_model()
    model.load()

    try:
        # Example 1: Simple RAG query (model can be None - will use instance model)
        print("\n=== Example 1: Simple RAG Query ===\n")
        result = rag.generate_with_context(
            query="What is machine learning?",
            model=None,  # Will use the model from rag instance
            top_k=3,
            max_tokens=150,
        )
        print(f"Query: What is machine learning?")
        print(f"\nAnswer: {result['response']}")
        print(f"\nSources: {result['sources']}")
        print(f"Number of sources: {result['num_sources']}")

        # Example 2: RAG query with filters (using instance model)
        print("\n=== Example 2: RAG Query with Filter ===\n")
        result = rag.generate_with_context(
            query="How does Python work?",
            model=None,  # Uses instance model automatically
            top_k=2,
            filters={"source": "Academic Fields"},  # Filter by source
            max_tokens=150,
        )
        print(f"Query: How does Python work?")
        print(f"\nAnswer: {result['response']}")
        print(f"\nSources: {result['sources']}")

        # Example 3: Just retrieval without generation
        print("\n=== Example 3: Retrieval Only ===\n")
        docs = rag.retrieve(
            query="What is artificial intelligence?",
            top_k=3,
        )
        print(f"Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. Question: {doc.get('question', 'N/A')}")
            print(f"   Answer: {doc.get('answer', 'N/A')[:100]}...")
            print(f"   Source: {doc.get('source', 'N/A')}")

        # Example 4: Custom context formatting
        print("\n=== Example 4: Custom Context Formatting ===\n")
        docs = rag.retrieve(query="What is STEM?", top_k=2)
        context = rag.format_context(docs, query="What is STEM?")
        print("Formatted context:")
        print(context)

    finally:
        # Cleanup
        rag.close()
        model.unload()
        print("\nCleanup complete.")


if __name__ == "__main__":
    main()
