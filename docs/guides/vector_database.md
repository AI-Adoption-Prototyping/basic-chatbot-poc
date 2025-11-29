# Vector Database / AI Data Infrastructure

## Overview

**Weaviate** is an open-source vector database designed for storing and searching high-dimensional embeddings. It enables semantic search and Retrieval-Augmented Generation (RAG) for AI applications. This project uses Weaviate as the core data infrastructure for storing and retrieving contextual information.

## Why Weaviate?

Weaviate was chosen for this project because it provides several critical features:

### Key Features

- **Hybrid Search**: Combines vector and keyword search for better results

  - Vector search for semantic similarity
  - Keyword search (BM25) for exact term matching
  - Configurable weighting between the two approaches

- **Filtering Capabilities**: Critical for role-based access control (RBAC)

  - Filter results by metadata fields (source, category, roles, etc.)
  - Enables limiting access by user roles in larger projects
  - Currently used for testing and demonstration purposes

- **GraphQL API**: Provides flexible querying capabilities

  - Type-safe queries
  - Easy to integrate with various frontend frameworks
  - Supports complex filtering and aggregation

- **Scalability**: Designed to handle large datasets in production environments

  - Horizontal scaling support
  - Efficient indexing and querying
  - Built-in persistence

- **Native Embeddings Support**: Can work with or without built-in vectorizers
  - Supports external embedding models (like SentenceTransformers)
  - Can use built-in vectorizers (e.g., text2vec-huggingface)
  - Flexible embedding pipeline

## Architecture in This Project

### Connection Configuration

Weaviate is configured to run as a Docker container with the following setup:

```yaml
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080" # HTTP REST API
      - "50051:50051" # gRPC API
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
```

### Data Schema

The project uses a simple schema optimized for question-answer pairs:

- **question**: The user's question (TEXT)
- **answer**: The answer to the question (TEXT)
- **context**: Additional context information (TEXT)
- **source**: Source identifier for filtering (TEXT)
- **Vector**: Dense embedding vector (generated from question)

The schema is defined in the ingestion script and uses external embeddings (SentenceTransformers) rather than Weaviate's built-in vectorizers.

### Query Strategies

The project supports multiple query strategies through the `WeaviateQuery` class:

1. **Vector Search**: Semantic similarity using embeddings
2. **Keyword Search**: BM25-based keyword matching
3. **Hybrid Search**: Weighted combination of vector and keyword search

See the [Querying Weaviate](query.md) guide for more details on query strategies.

## Setup and Configuration

### Starting Weaviate

Weaviate runs as a Docker container. To start it:

```bash
# Start Weaviate (runs in the background)
docker-compose up -d

# Verify it's running
docker-compose ps

# Check logs if needed
docker-compose logs weaviate
```

### Configuration Options

Weaviate can be configured through environment variables:

- `WEAVIATE_HOST`: Server hostname (default: `localhost`)
- `WEAVIATE_PORT`: HTTP port (default: `8080`)
- `WEAVIATE_GRPC_PORT`: gRPC port (default: `50051`)
- `WEAVIATE_COLLECTION_NAME`: Collection name for RAG entries (default: `RAGEntry`)

### Data Ingestion

Data is loaded into Weaviate using the ingestion script:

```bash
python ingest/ingest_weaviate.py ingest/rag_dataset_500.json
```

The ingestion process:

1. Creates the collection schema if it doesn't exist
2. Loads data from JSON files
3. Generates embeddings using SentenceTransformers
4. Inserts data with vectors into Weaviate

## Filtering and Access Control

### Current Implementation

Currently, filtering is used for:

- Testing different data sources
- Demonstrating filtering capabilities
- Preparing for role-based access control

### Future: Role-Based Access Control

The filtering capability is a key driving force in choosing Weaviate. In larger projects, it will be used to:

- Limit access by user roles
- Filter results based on permissions
- Implement multi-tenant data isolation
- Control data visibility based on organizational structure

### Example Filtering

```python
# Filter by source
filters = {"source": "IT Help"}

# Filter by multiple criteria
filters = {
    "source": "Academic Fields",
    "allowed_roles": ["student", "faculty"]
}
```

## Integration with RAG Pipeline

Weaviate fits into the RAG pipeline as follows:

1. **Ingestion**: Data is loaded and embedded into Weaviate
2. **Query**: User questions are converted to embeddings and searched
3. **Retrieval**: Relevant documents are retrieved with filtering
4. **Generation**: Retrieved context is passed to the LLM for answer generation

The `WeaviateRAG` class encapsulates this integration, providing a clean interface for the RAG system.

## Alternatives Considered

Other vector databases that were considered:

- **FAISS**: Facebook's similarity search library

  - Pros: Fast, efficient
  - Cons: No built-in filtering, requires more manual setup

- **Chroma**: Open-source embedding database

  - Pros: Simple, Python-native
  - Cons: Less mature filtering capabilities

- **Qdrant**: Vector similarity search engine

  - Pros: Good performance, filtering support
  - Cons: Less mature ecosystem

- **Pinecone**: Managed vector database
  - Pros: Fully managed, scalable
  - Cons: Requires cloud service, costs money

Weaviate was chosen for its combination of features, open-source nature, and strong filtering capabilities that align with the project's future RBAC requirements.

## Best Practices

### Data Organization

- Use clear source identifiers for filtering
- Include relevant metadata for each entry
- Structure data consistently for predictable queries

### Query Optimization

- Use hybrid search for best results
- Adjust `top_k` based on your needs (typically 3-5 for RAG)
- Use filters to narrow results when appropriate

### Performance

- Batch insertions for better performance
- Use appropriate embedding models for your use case
- Monitor query performance and adjust as needed

## Troubleshooting

### Connection Issues

If you can't connect to Weaviate:

1. Verify Docker container is running: `docker-compose ps`
2. Check port availability: `curl http://localhost:8080/v1/meta`
3. Review logs: `docker-compose logs weaviate`

### Query Performance

If queries are slow:

1. Check the number of documents in the collection
2. Verify embedding model is appropriate for your data
3. Consider using filters to reduce search space
4. Monitor Weaviate resource usage

### Data Issues

If data isn't appearing in queries:

1. Verify data was ingested successfully
2. Check collection name matches configuration
3. Ensure embeddings were generated correctly
4. Test with simple queries first

## Resources

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Weaviate Python Client](https://weaviate.io/developers/weaviate/client-libraries/python)
- [Weaviate Query Examples](https://weaviate.io/developers/weaviate/api/graphql)
