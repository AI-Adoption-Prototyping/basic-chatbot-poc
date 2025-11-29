# Querying Weaviate

## Overview: WeaviateQuery Class

The `WeaviateQuery` class provides a Python interface for querying a Weaviate vector database. It supports multiple search strategies to retrieve relevant context for RAG applications.

### Supported Search Strategies

- **Vector Search**: Semantic similarity using embeddings

  - Best for natural language queries where meaning matters more than exact keywords
  - Uses cosine similarity on dense vector embeddings

- **Keyword Search**: BM25-based keyword matching

  - Useful for exact term matching and traditional information retrieval
  - Effective when specific keywords are important

- **Hybrid Search**: Combines semantic and keyword search

  - Balances semantic meaning and keyword relevance
  - Typically provides better recall than either method alone
  - Uses a weighted combination of vector and keyword scores

- **Metadata Filtering**: Enables category-based or source-based filtering
  - Filter results by source, category, or other metadata fields
  - Critical for role-based access control (RBAC) implementations
  - Can be combined with any search strategy

## Why These Methods Exist

In Retrieval-Augmented Generation (RAG), retrieving relevant context before passing it to an LLM is critical for accurate responses. Different query strategies improve precision and recall in different scenarios:

- **Vector Search**: Best for natural language queries where synonyms and semantic relationships matter
- **Keyword Search**: Useful when exact term matching is important or when dealing with technical terminology
- **Hybrid Search**: Provides the best of both worlds, improving overall retrieval quality

## Where It Fits in the RAG Pipeline

The query process fits into the RAG pipeline as follows:

1. **User asks a question** → Input received by the application
2. **Generate query embedding** → Use SentenceTransformer to create a vector representation
3. **Query Weaviate** → Search for relevant documents using one of the search strategies
4. **Retrieve context** → Get top-k most relevant documents
5. **Pass to LLM** → Include retrieved context in the prompt for final answer generation

This pipeline ensures that the LLM has access to relevant, contextual information rather than relying solely on its training data.

## Common Alternatives

If you're considering alternatives to Weaviate, here are some options:

- **Vector Databases**: FAISS, Chroma, Qdrant, Milvus, Pinecone
- **Keyword Search**: Elasticsearch, OpenSearch, Solr
- **Hybrid Search**: Qdrant (native hybrid), Elasticsearch with dense vectors

Weaviate was chosen for this project because it provides native support for hybrid search, filtering capabilities (essential for RBAC), and a flexible GraphQL API.

## Docker ARM Issue and Solution

### The Problem

When running HuggingFace inference containers on ARM64 systems (e.g., Apple Silicon Macs), Docker may fail with:

```
no matching manifest for linux/arm64/v8 in the manifest list entries
```

This occurs because some HuggingFace images do not provide native ARM builds.

### Solutions

#### Option 1: Force AMD64 Emulation

Add `platform: linux/amd64` to your `docker-compose.yml`:

```yaml
services:
  transformers:
    image: ghcr.io/huggingface/text-embeddings-inference:latest
    platform: linux/amd64
```

**Pros**: Works with any Docker image  
**Cons**: Uses QEMU emulation, which is slower than native execution

#### Option 2: Use HuggingFace API (Recommended for ARM)

Switch Weaviate to use the `text2vec-huggingface` module with the HuggingFace API:

```yaml
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    environment:
      ENABLE_MODULES: "text2vec-huggingface"
      HUGGINGFACE_APIKEY: "your_huggingface_api_key"
```

**Pros**:

- Works natively on ARM without emulation
- No need to run a separate inference container
- Better performance on ARM systems

**Cons**: Requires a HuggingFace API key (free tier available)

### Recommended Approach

For ARM-based systems (Apple Silicon), use **Weaviate with text2vec-huggingface module** and the HuggingFace API. This provides the best combination of simplicity and performance.

For this project, we use **local embeddings with SentenceTransformers** (`all-MiniLM-L6-v2`), which works natively on all platforms without requiring Docker containers or API keys.

## Design Benefits

The `WeaviateQuery` class design provides several advantages:

- **Encapsulation**: All query logic is contained in one class, making it easy to maintain and test
- **Flexibility**: Supports multiple search strategies, allowing you to choose the best approach for each use case
- **Extensibility**: Easy to extend with new filters, query types, or search strategies
- **Integration**: Works seamlessly with SentenceTransformers for local embeddings
- **Type Safety**: Uses Weaviate's v4 Python client with proper type hints

This design makes it straightforward to experiment with different retrieval strategies and optimize for your specific use case.
