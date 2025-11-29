#!/usr/bin/env python3
"""
Simple CLI for querying Weaviate with filters.
Usage examples:
    python query_cli.py --query "machine learning" --limit 5
    python query_cli.py --query "python" --source "Academic Fields"
    python query_cli.py --query "AI" --method hybrid --limit 10
"""
import argparse
from query_weaviate import WeaviateQuery, filter_by_source, filter_by_source_contains
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description="Query Weaviate vector database")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text to search for",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["vector", "keyword", "hybrid"],
        default="vector",
        help="Search method (default: vector)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Filter by exact source name",
    )
    parser.add_argument(
        "--source-contains",
        type=str,
        help="Filter where source contains text",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid search alpha (0=keyword, 1=vector, default: 0.5)",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Show distance/score metadata",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)",
    )

    args = parser.parse_args()

    # Initialize query client
    query_client = WeaviateQuery()
    embedding_model = SentenceTransformer(args.model)

    try:
        # Build filter if specified
        filter_obj = None
        if args.source:
            filter_obj = filter_by_source(args.source)
        elif args.source_contains:
            filter_obj = filter_by_source_contains(args.source_contains)

        # Execute query based on method
        if args.method == "vector":
            results = query_client.query_by_vector(
                query_text=args.query,
                embedding_model=embedding_model,
                limit=args.limit,
                filters=filter_obj,
                return_metadata=args.show_metadata,
            )
        elif args.method == "keyword":
            results = query_client.query_by_keyword(
                query_text=args.query,
                limit=args.limit,
                filters=filter_obj,
            )
        elif args.method == "hybrid":
            results = query_client.hybrid_search(
                query_text=args.query,
                embedding_model=embedding_model,
                limit=args.limit,
                alpha=args.alpha,
                filters=filter_obj,
                return_metadata=args.show_metadata,
            )

        # Display results
        print(f"\n=== Found {len(results)} results ===\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. Question: {result['question']}")
            print(f"   Answer: {result['answer'][:150]}...")
            print(f"   Source: {result['source']}")
            if args.show_metadata:
                if "distance" in result:
                    print(f"   Distance: {result['distance']:.4f}")
                if "score" in result:
                    print(f"   Score: {result['score']:.4f}")
            print()

    finally:
        query_client.close()


if __name__ == "__main__":
    main()
