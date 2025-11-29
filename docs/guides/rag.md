# Retrieval-Augmented Generation (RAG)

## Overview

**Retrieval-Augmented Generation (RAG)** is a technique that enhances large language models by providing them with relevant context retrieved from a knowledge base. This project implements RAG using Weaviate as the vector database and SentenceTransformers for embeddings.

## What is RAG?

RAG combines two key components:

1. **Retrieval**: Finding relevant information from a knowledge base
2. **Augmentation**: Providing that information as context to the LLM
3. **Generation**: The LLM generates responses based on the retrieved context

This approach enables LLMs to:

- Answer questions about information not in their training data
- Provide up-to-date information
- Cite sources for their answers
- Reduce hallucinations by grounding responses in retrieved context

## How RAG Works

### The RAG Pipeline

1. **User Query**: User asks a question
2. **Query Embedding**: Convert query to vector embedding using SentenceTransformers
3. **Vector Search**: Search Weaviate for similar documents using the query embedding
4. **Context Retrieval**: Retrieve top-k most relevant documents
5. **Context Formatting**: Format retrieved documents into context string
6. **Prompt Construction**: Combine query, context, and instructions into prompt
7. **LLM Generation**: LLM generates response based on the prompt
8. **Response Return**: Return answer with source citations

### Visual Flow

```
User Query
    ↓
Query Embedding (SentenceTransformers)
    ↓
Vector Search (Weaviate)
    ↓
Retrieve Top-K Documents
    ↓
Format Context
    ↓
Construct Prompt
    ↓
LLM Generation
    ↓
Response + Sources
```

## Implementation in This Project

### Architecture

The RAG system is implemented using a base class pattern:

- **BaseRAG**: Abstract interface defining RAG operations
- **WeaviateRAG**: Concrete implementation using Weaviate

This separation allows for:

- Easy testing of components
- Swapping implementations if needed
- Clear separation of concerns

### Key Components

#### WeaviateRAG Class

The `WeaviateRAG` class handles:

- **Connection Management**: Manages Weaviate client connections
- **Embedding Generation**: Uses SentenceTransformers for embeddings
- **Query Execution**: Performs vector/keyword/hybrid searches
- **Context Formatting**: Formats retrieved documents for LLM
- **Model Integration**: Integrates with LLM models for generation

#### Query Strategies

The system supports multiple query strategies:

- **Vector Search**: Semantic similarity using embeddings
- **Keyword Search**: BM25-based keyword matching
- **Hybrid Search**: Combines vector and keyword search

See the [Querying Weaviate](query.md) guide for details.

### Prompt Template

The system uses a prompt template to instruct the LLM:

```text
Use the following context to answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{retrieved_context}

Question: {user_query}

Answer:
```

This ensures:

- Responses are grounded in retrieved context
- The LLM knows when information is insufficient
- Clear separation between context and query

## Usage Examples

### Basic RAG Query

```python
from rag import WeaviateRAG
from models import GGUFModel

# Initialize RAG system
rag = WeaviateRAG(model_class=GGUFModel)

# Generate response with context
result = rag.generate_with_context(
    query="What is machine learning?",
    top_k=3,
    max_tokens=150,
)

print(result['response'])
print(result['sources'])
```

### RAG with Filtering

```python
# Filter by source
result = rag.generate_with_context(
    query="How does Python work?",
    top_k=2,
    filters={"source": "Academic Fields"},
    max_tokens=150,
)
```

### Retrieval Only

```python
# Just retrieve documents without generation
docs = rag.retrieve(
    query="What is artificial intelligence?",
    top_k=3,
)

for doc in docs:
    print(doc['question'])
    print(doc['answer'])
```

## Configuration

### Top-K Selection

The `top_k` parameter controls how many documents to retrieve:

- **Small (1-3)**: Focused, concise context
- **Medium (3-5)**: Balanced coverage
- **Large (5-10)**: Comprehensive but may include irrelevant info

**Recommendation**: Start with 3-5 for most use cases.

### Context Window

The retrieved context must fit within the LLM's context window:

- **Model Context**: 4096 tokens (configurable)
- **Query + Instructions**: ~50-100 tokens
- **Available for Context**: ~4000 tokens
- **Documents**: Varies, typically 100-500 tokens each

Plan your `top_k` accordingly to fit within the context window.

## Advantages of RAG

### Up-to-Date Information

- LLMs can access information not in their training data
- Can be updated by adding new documents to the knowledge base
- No need to retrain the model

### Source Attribution

- Can cite sources for answers
- Enables fact-checking
- Builds user trust

### Reduced Hallucinations

- Responses are grounded in retrieved context
- LLM is instructed to say when information is insufficient
- Less likely to make up facts

### Domain Adaptation

- Easy to adapt to new domains by adding domain-specific documents
- No fine-tuning required
- Can handle multiple domains simultaneously

## Limitations

### Retrieval Quality

- Quality depends on retrieval quality
- Poor retrieval = poor responses
- Requires good embeddings and search strategy

### Context Window Limits

- Limited by LLM context window
- May need to truncate or summarize long documents
- Trade-off between coverage and detail

### Latency

- Additional latency from retrieval step
- Embedding generation adds time
- Vector search adds time

### Data Quality

- Requires high-quality source documents
- Garbage in = garbage out
- Need to maintain and update knowledge base

## Best Practices

### Document Quality

- Use clear, well-structured documents
- Include relevant metadata (source, category, etc.)
- Keep documents focused and concise

### Query Optimization

- Use hybrid search for best results
- Adjust `top_k` based on your needs
- Use filters to narrow results when appropriate

### Prompt Engineering

- Clear instructions for the LLM
- Explicit format for context
- Instructions for handling insufficient information

### Evaluation

- Test with diverse queries
- Evaluate retrieval quality separately
- Monitor response quality in production

## Advanced Topics

### Chunking Strategies

For longer documents, chunking is critical:

- **Fixed-size chunks**: Simple but may split context
- **Semantic chunks**: Better but more complex
- **Overlapping chunks**: Preserves context boundaries

This project uses simple question-answer pairs, but production systems often require chunking.

### Re-ranking

After initial retrieval, re-ranking can improve results:

- Use a more powerful model for re-ranking
- Re-rank top-k results for better ordering
- Trade-off between quality and latency

### Multi-hop Retrieval

For complex queries, multiple retrieval steps:

1. Initial retrieval
2. Use retrieved context to refine query
3. Second retrieval with refined query
4. Combine results

### Conversation History

This project includes conversation history:

- Maintains context across turns
- Can use history to improve retrieval
- Enables follow-up questions

## Troubleshooting

### Poor Retrieval Results

If retrieval isn't finding relevant documents:

1. Check embedding quality
2. Verify documents were ingested correctly
3. Try different search strategies (hybrid vs. vector)
4. Adjust `top_k` value
5. Review query formulation

### Irrelevant Context

If retrieved context is irrelevant:

1. Improve document quality and structure
2. Use filters to narrow search space
3. Adjust embedding model if needed
4. Consider re-ranking

### LLM Ignoring Context

If the LLM isn't using retrieved context:

1. Review prompt template
2. Make instructions more explicit
3. Check context formatting
4. Verify context is actually relevant

### Slow Performance

If RAG is too slow:

1. Optimize embedding generation (batch processing)
2. Use appropriate `top_k` (not too large)
3. Consider caching embeddings
4. Optimize Weaviate queries
