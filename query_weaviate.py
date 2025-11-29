import time
import weaviate
import weaviate.classes as wvc
from typing import Optional, List, Dict, Any
import json
from config import get_settings


class WeaviateQuery:
    """Query interface for Weaviate vector database."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize connection to Weaviate.

        Args:
            host: Weaviate host (defaults to config)
            port: HTTP port (defaults to config)
            grpc_port: gRPC port (defaults to config)
            collection_name: Collection name (defaults to config)
        """
        settings = get_settings()

        host = host or settings.weaviate_host
        port = port or settings.weaviate_port
        grpc_port = grpc_port or settings.weaviate_grpc_port
        self.collection_name = collection_name or settings.weaviate_collection_name

        self.client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=grpc_port,
            skip_init_checks=True,  # Skip gRPC health check to use HTTP-only mode if needed
        )

    def query_by_vector(
        self,
        query_text: str,
        embedding_model,
        limit: int = 5,
        filters: Optional[wvc.query.Filter] = None,
        return_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Query Weaviate using vector similarity search.

        Args:
            query_text: Text to search for
            embedding_model: SentenceTransformer model to encode the query
            limit: Maximum number of results to return
            filters: Optional filter conditions
            return_metadata: Whether to return metadata (distance, certainty)

        Returns:
            List of matching results with properties and optionally metadata
        """
        # Check if collection exists
        if not self.client.collections.exists(self.collection_name):
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist in Weaviate. "
                f"Please ingest data first using: python ingest/ingest_weaviate.py <data_file>"
            )

        # Generate embedding for the query (timed)
        embedding_start = time.time()
        query_vector = embedding_model.encode(query_text).tolist()
        embedding_time = time.time() - embedding_start

        # Get collection
        collection = self.client.collections.get(self.collection_name)

        # Build query - filters are passed directly as a parameter
        query_start = time.time()
        query_result = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            filters=filters,
            return_metadata=(
                wvc.query.MetadataQuery(distance=True) if return_metadata else None
            ),
        )
        query_time = time.time() - query_start

        # Format results - objects are directly accessible
        results = []
        for obj in query_result.objects:
            result = {
                "question": obj.properties.get("question", ""),
                "answer": obj.properties.get("answer", ""),
                "context": obj.properties.get("context", ""),
                "source": obj.properties.get("source", ""),
            }
            if return_metadata and obj.metadata:
                result["distance"] = obj.metadata.distance
            results.append(result)
        
        # Store timing info in results metadata
        if results:
            results[0]["_timing"] = {
                "embedding_time": embedding_time,
                "query_time": query_time,
                "total_retrieval_time": embedding_time + query_time,
            }

        return results

    def query_by_keyword(
        self,
        query_text: str,
        limit: int = 5,
        filters: Optional[wvc.query.Filter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query Weaviate using BM25 keyword search.

        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            filters: Optional filter conditions

        Returns:
            List of matching results
        """
        # Check if collection exists
        if not self.client.collections.exists(self.collection_name):
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist in Weaviate. "
                f"Please ingest data first using: python ingest/ingest_weaviate.py <data_file>"
            )

        collection = self.client.collections.get(self.collection_name)

        # Build query - filters are passed directly as a parameter
        query_result = collection.query.bm25(
            query=query_text,
            limit=limit,
            filters=filters,
        )

        # Format results - objects are directly accessible
        results = []
        for obj in query_result.objects:
            results.append(
                {
                    "question": obj.properties.get("question", ""),
                    "answer": obj.properties.get("answer", ""),
                    "context": obj.properties.get("context", ""),
                    "source": obj.properties.get("source", ""),
                }
            )

        return results

    def hybrid_search(
        self,
        query_text: str,
        embedding_model,
        limit: int = 5,
        alpha: float = 0.5,
        filters: Optional[wvc.query.Filter] = None,
        return_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and keyword search.

        Args:
            query_text: Text to search for
            embedding_model: SentenceTransformer model to encode the query
            limit: Maximum number of results to return
            alpha: Weight between vector (1.0) and keyword (0.0) search
            filters: Optional filter conditions
            return_metadata: Whether to return metadata

        Returns:
            List of matching results
        """
        # Check if collection exists
        if not self.client.collections.exists(self.collection_name):
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist in Weaviate. "
                f"Please ingest data first using: python ingest/ingest_weaviate.py <data_file>"
            )
        # Generate embedding for the query
        query_vector = embedding_model.encode(query_text).tolist()

        collection = self.client.collections.get(self.collection_name)

        # Build hybrid query - filters and metadata specified in initial call
        query_result = collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=alpha,
            limit=limit,
            filters=filters,
            return_metadata=(
                wvc.query.MetadataQuery(distance=True, score=True)
                if return_metadata
                else None
            ),
        )

        # Format results - objects are directly accessible
        results = []
        for obj in query_result.objects:
            result = {
                "question": obj.properties.get("question", ""),
                "answer": obj.properties.get("answer", ""),
                "context": obj.properties.get("context", ""),
                "source": obj.properties.get("source", ""),
            }
            if return_metadata and obj.metadata:
                if hasattr(obj.metadata, "distance"):
                    result["distance"] = obj.metadata.distance
                if hasattr(obj.metadata, "score"):
                    result["score"] = obj.metadata.score
            results.append(result)

        return results

    def get_all(
        self,
        limit: int = 10,
        filters: Optional[wvc.query.Filter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all objects (optionally filtered).

        Args:
            limit: Maximum number of results
            filters: Optional filter conditions

        Returns:
            List of results
        """
        collection = self.client.collections.get(self.collection_name)

        # Fetch objects with filters passed directly as parameter
        query_result = collection.query.fetch_objects(
            limit=limit,
            filters=filters,
        )

        # Format results - objects are directly accessible
        results = []
        for obj in query_result.objects:
            results.append(
                {
                    "question": obj.properties.get("question", ""),
                    "answer": obj.properties.get("answer", ""),
                    "context": obj.properties.get("context", ""),
                    "source": obj.properties.get("source", ""),
                }
            )

        return results

    def close(self):
        """Close the Weaviate connection."""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
                print("Weaviate connection closed")
        except Exception as e:
            print(f"Error closing Weaviate connection: {e}")


# Example filter builders
def filter_by_source(source: str) -> wvc.query.Filter:
    """Filter by exact source match."""
    return wvc.query.Filter.by_property("source").equal(source)


def filter_by_source_contains(text: str) -> wvc.query.Filter:
    """Filter where source contains text."""
    return wvc.query.Filter.by_property("source").like(f"*{text}*")


def filter_by_question_contains(text: str) -> wvc.query.Filter:
    """Filter where question contains text."""
    return wvc.query.Filter.by_property("question").like(f"*{text}*")


def filter_by_multiple_sources(sources: List[str]) -> wvc.query.Filter:
    """Filter by multiple sources (OR condition)."""
    return wvc.query.Filter.any_of(
        [wvc.query.Filter.by_property("source").equal(src) for src in sources]
    )


def combine_filters(
    filter1: wvc.query.Filter, filter2: wvc.query.Filter
) -> wvc.query.Filter:
    """Combine two filters with AND logic."""
    return wvc.query.Filter.all_of([filter1, filter2])


# Example usage
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # Initialize query client
    query_client = WeaviateQuery()

    # Load embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    try:
        print("=== Example 1: Vector Search ===\n")
        results = query_client.query_by_vector(
            query_text="What is machine learning?",
            embedding_model=embedding_model,
            limit=3,
        )
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['question']}")
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Source: {result['source']}\n")

        print("\n=== Example 2: Vector Search with Filter ===\n")
        # Filter by source
        source_filter = filter_by_source("example_source")  # Replace with actual source
        results = query_client.query_by_vector(
            query_text="machine learning",
            embedding_model=embedding_model,
            limit=3,
            filters=source_filter,
        )
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['question']}")
            print(f"   Source: {result['source']}\n")

        print("\n=== Example 3: Keyword Search ===\n")
        results = query_client.query_by_keyword(
            query_text="python programming",
            limit=3,
        )
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['question']}")
            print(f"   Answer: {result['answer'][:100]}...\n")

        print("\n=== Example 4: Hybrid Search ===\n")
        results = query_client.hybrid_search(
            query_text="artificial intelligence",
            embedding_model=embedding_model,
            limit=3,
            alpha=0.7,  # More weight on vector search
        )
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['question']}")
            print(f"   Answer: {result['answer'][:100]}...\n")

        print("\n=== Example 5: Get All with Filter ===\n")
        # Filter where source contains specific text
        filter_contains = filter_by_source_contains(
            "example"
        )  # Replace with actual text
        results = query_client.get_all(
            limit=5,
            filters=filter_contains,
        )
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['question']}")
            print(f"   Source: {result['source']}\n")

        print("\n=== Example 6: Complex Filter (Multiple Sources) ===\n")
        sources_filter = filter_by_multiple_sources(
            ["source1", "source2"]
        )  # Replace with actual sources
        results = query_client.query_by_vector(
            query_text="technology",
            embedding_model=embedding_model,
            limit=3,
            filters=sources_filter,
        )
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['question']}")
            print(f"   Source: {result['source']}\n")

    finally:
        query_client.close()
        print("Connection closed.")
