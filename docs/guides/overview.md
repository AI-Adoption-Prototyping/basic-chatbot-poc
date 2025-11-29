# Overview

This project is a proof of concept demonstrating that a modest machine can run Retrieval-Augmented Generation (RAG) efficiently on a laptop while maintaining high performance with filtering and session history capabilities. While not all best practices were followed, the key components are implemented properly.

## Architectural Choices

### Inference Engine

**llama.cpp** is an open-source C++ implementation for inference of large language models like LLaMA and Mistral. It's optimized for running on CPUs without requiring GPUs or heavy frameworks. The library supports quantized model formats (such as GGUF), enabling efficient, low-resource deployment on desktops, servers, and even edge devices.

### Storage and Inference Format

**GGUF** (Generalized GGML Unified Format) is a binary format designed for efficient inference of large language models on CPU-based systems. It supports multiple quantization levels to reduce memory usage and improve inference speed. This format enables deployment without GPU dependencies, making it ideal for lightweight, on-premise, and resource-constrained environments.

### Vector Database / AI Data Infrastructure

**Weaviate** is an open-source vector database designed for storing and searching high-dimensional embeddings. It enables semantic search and Retrieval-Augmented Generation (RAG) for AI applications. Key features include:

- **Hybrid queries**: Combines vector and keyword search for better results
- **Filtering capabilities**: Critical to our design, enabling role-based access control (RBAC) for limiting access by user roles
- **GraphQL API**: Provides flexible querying capabilities
- **Scalability**: Designed to handle large datasets in production environments

The filtering capability is a key driving force in choosing Weaviate, as it will be used to limit access by role in larger projects. Currently, it's used for testing and demonstration purposes.

### Sentence Embedding Model / Semantic Search

**all-MiniLM-L6-v2** is a lightweight transformer-based model from the Sentence Transformers library. It's designed to generate dense vector embeddings for sentences and short texts. This model is widely used for:

- Semantic similarity tasks
- Text clustering
- Retrieval-Augmented Generation (RAG)

It balances speed, accuracy, and resource efficiency, making it ideal for this project. The model is compact, high-quality for semantic tasks, resource-friendly, and versatile—supporting multilingual text and being tunable for specific domains.

### Large Language Model

**TheBloke/Mistral-7B-Instruct-v0.2-GGUF** is a quantized version of the Mistral 7B instruction-tuned model in GGUF format. It's optimized for efficient inference on CPUs without requiring GPUs or heavy frameworks.

The **mistral-7b-instruct-v0.2.Q4_K_M.gguf** variant is recommended as it provides an excellent balance of quality, performance, and size. If you need to experiment with different performance characteristics, you can explore [other quantization options](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) or alternative models.

### Retrieval-Augmented Generation (RAG)

**RAG** context is queried from **Weaviate** using filters and vector search. The system strictly uses the RAG data for replying based on the prompt structure. The initial prompt template (which may still be in use) is:

```text
Use the following context to answer the question. If the context doesn't contain enough information to answer the question, say so.
```

This ensures that responses are grounded in the provided context, improving accuracy and reducing hallucinations.

### Separation of Responsibilities

The codebase is organized by function to allow for easier maintenance and testing. For example, the AI model implementation should not know about the RAG implementation details. This separation enables:

- Independent testing of components
- Easier swapping of implementations
- Better code organization
- Reduced coupling between modules

### Base Classes/Interfaces

As with any project, you shouldn't start from scratch if good solid components exist. One of the hardest parts of architecture is picking a core library. To minimize this risk, the project uses an interface/base class pattern. Key components have base classes that define their interfaces:

- **BaseModel**: Abstract interface for LLM models
- **BaseRAG**: Abstract interface for RAG implementations

Since tools have been selected for each category, these base classes may need to be enhanced as the project evolves. They are not considered locked at this point and can be extended as needed.

### Simple Documents

The project loads simple documents for question-answer pairs without concern for chunking strategies. While chunking is critical to quality in production RAG systems, this project focuses on demonstrating the core RAG pipeline. Other projects will cover chunking basics in more detail.

RAG data is loaded using the script in the **ingest** directory, which processes JSON files containing question-answer pairs.

### Helper Scripts

The project includes several helper scripts for development and testing:

- **query_cli.py** - Command-line interface to query **Weaviate** directly
- **ingest/ingest_weaviate.py** - Script to load data into **Weaviate** from JSON files
- **rag_example.py** - Small example script demonstrating RAG usage
- **query_weaviate.py** - Runnable script with examples of querying Weaviate
