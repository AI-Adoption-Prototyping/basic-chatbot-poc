# Helper/Example Scripts

## Overview

This project includes several helper scripts for development, testing, and data management. These scripts provide command-line interfaces and examples for common tasks.

## Available Scripts

### Data Ingestion

#### `ingest/ingest_weaviate.py`

Loads data from JSON files into Weaviate with embeddings.

**Usage:**

```bash
python ingest/ingest_weaviate.py <data_file>
```

**Example:**

```bash
python ingest/ingest_weaviate.py ingest/rag_dataset_500.json
python ingest/ingest_weaviate.py ingest/it_help_qna.json
```

**What it does:**

1. Connects to Weaviate
2. Creates collection schema if it doesn't exist
3. Loads data from JSON file
4. Generates embeddings using SentenceTransformers
5. Inserts data with vectors into Weaviate

**Data Format:**

The script expects JSON files with the following structure:

```json
[
  {
    "question": "What is machine learning?",
    "answer": "Machine learning is a subset of AI...",
    "context": "Additional context information",
    "source": "Academic Fields"
  }
]
```

**Features:**

- Automatic schema creation
- Batch insertion for performance
- Progress reporting (every 50 entries)
- Command-line argument for data file path

### Query Scripts

#### `query_cli.py`

Command-line interface for querying Weaviate directly.

**Usage:**

```bash
python query_cli.py --query "your query" [options]
```

**Options:**

- `--query`: Query text to search for (required)
- `--method`: Search method - `vector`, `keyword`, or `hybrid` (default: `vector`)
- `--limit`: Maximum number of results (default: `5`)
- `--source`: Filter by exact source name
- `--source-contains`: Filter where source contains text
- `--alpha`: Hybrid search alpha (0=keyword, 1=vector, default: `0.5`)

**Examples:**

```bash
# Simple vector search
python query_cli.py --query "machine learning" --limit 5

# Keyword search with filter
python query_cli.py --query "python" --method keyword --source "Academic Fields"

# Hybrid search
python query_cli.py --query "AI" --method hybrid --limit 10 --alpha 0.7

# Filter by source contains
python query_cli.py --query "email" --source-contains "IT"
```

**Output:**

The script displays:
- Query information
- Search method used
- Number of results
- Each result with question, answer, source, and similarity score

**Use Cases:**

- Testing Weaviate queries
- Debugging search results
- Exploring data in the database
- Comparing different search methods

#### `query_weaviate.py`

Runnable script with examples of querying Weaviate.

**Usage:**

```bash
python query_weaviate.py
```

**What it does:**

This script demonstrates various ways to query Weaviate:

1. **Vector Search Examples**: Semantic similarity queries
2. **Keyword Search Examples**: BM25-based queries
3. **Hybrid Search Examples**: Combined vector and keyword search
4. **Filtering Examples**: Queries with metadata filters

**Features:**

- Multiple query examples
- Demonstrates different search strategies
- Shows filtering capabilities
- Educational examples for learning the API

**Use Cases:**

- Learning how to use WeaviateQuery class
- Understanding different search methods
- Testing query functionality
- Reference for implementing queries

### RAG Examples

#### `rag_example.py`

Example script demonstrating RAG system usage.

**Usage:**

```bash
python rag_example.py
```

**What it demonstrates:**

1. **Simple RAG Query**: Basic question-answering with context
2. **RAG with Filters**: Filtering by source or other metadata
3. **Retrieval Only**: Getting documents without generation
4. **Custom Context Formatting**: Formatting retrieved context

**Example Output:**

```
=== Example 1: Simple RAG Query ===

Query: What is machine learning?

Answer: Machine learning is a subset of artificial intelligence...

Sources: ['Academic Fields', 'Tech Basics']
Number of sources: 2
```

**Features:**

- Complete RAG pipeline examples
- Shows integration with LLM models
- Demonstrates filtering capabilities
- Proper cleanup and resource management

**Use Cases:**

- Learning RAG implementation
- Testing RAG functionality
- Understanding the RAG pipeline
- Reference for building RAG applications

### Model Loading

#### `load_model.py`

Example script for loading and using GGUF models.

**Usage:**

```bash
python load_model.py
```

**What it does:**

1. Downloads model from Hugging Face (if not cached)
2. Loads GGUF model using llama-cpp-python
3. Demonstrates model configuration
4. Shows basic usage

**Features:**

- Automatic model download
- Model configuration examples
- Basic inference example
- Resource usage information

**Use Cases:**

- Testing model loading
- Understanding model configuration
- Learning llama-cpp-python usage
- Reference for model setup

## Script Organization

### Directory Structure

```
.
├── ingest/
│   └── ingest_weaviate.py    # Data ingestion script
├── query_cli.py               # CLI for querying Weaviate
├── query_weaviate.py          # Query examples
├── rag_example.py             # RAG usage examples
└── load_model.py              # Model loading example
```

### Common Patterns

All scripts follow similar patterns:

1. **Configuration Loading**: Use `config.get_settings()` for configuration
2. **Error Handling**: Proper try/except blocks and cleanup
3. **Resource Management**: Close connections and unload models
4. **Progress Reporting**: Show progress for long-running operations
5. **Command-Line Arguments**: Use `argparse` for CLI scripts

## Best Practices

### Running Scripts

1. **Activate Virtual Environment**: Always use the project's virtual environment
   ```bash
   source .venv/bin/activate
   ```

2. **Check Prerequisites**: Ensure Weaviate is running for query scripts
   ```bash
   docker-compose ps
   ```

3. **Verify Data**: Ensure data is ingested before querying
   ```bash
   python ingest/ingest_weaviate.py ingest/rag_dataset_500.json
   ```

### Development Workflow

1. **Data Ingestion**: Use `ingest_weaviate.py` to load test data
2. **Query Testing**: Use `query_cli.py` to test queries
3. **RAG Testing**: Use `rag_example.py` to test RAG functionality
4. **Model Testing**: Use `load_model.py` to test model loading

### Debugging

- Use `--verbose` flags where available
- Check Weaviate logs: `docker-compose logs weaviate`
- Verify data in Weaviate using `query_cli.py`
- Test individual components before integration

## Extending Scripts

### Adding New Scripts

When creating new scripts:

1. **Follow Naming Conventions**: Use descriptive names
2. **Add Documentation**: Include docstrings and comments
3. **Use Configuration**: Load settings from config
4. **Handle Errors**: Proper error handling and cleanup
5. **Add Examples**: Include usage examples in docstrings

### Example Template

```python
#!/usr/bin/env python3
"""
Script description.

Usage:
    python script_name.py [options]
"""
import argparse
from config import get_settings

def main():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--option", type=str, help="Option description")
    args = parser.parse_args()
    
    settings = get_settings()
    
    try:
        # Script logic here
        pass
    finally:
        # Cleanup
        pass

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

#### Import Errors

If you encounter import errors:

```bash
# Ensure you're in the project root
cd /path/to/basic-bot

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Weaviate Connection Errors

If scripts can't connect to Weaviate:

```bash
# Check if Weaviate is running
docker-compose ps

# Start Weaviate if needed
docker-compose up -d

# Check Weaviate logs
docker-compose logs weaviate
```

#### Model Loading Errors

If model loading fails:

1. Check internet connection (for download)
2. Verify Hugging Face access
3. Check available disk space
4. Verify model filename in configuration

#### Data Not Found

If queries return no results:

1. Verify data was ingested: `python query_cli.py --query "test"`
2. Check collection name matches configuration
3. Verify data file format is correct
4. Check Weaviate logs for errors

## Resources

- [Python argparse Documentation](https://docs.python.org/3/library/argparse.html)
- [Weaviate Python Client](https://weaviate.io/developers/weaviate/client-libraries/python)
- [Sentence Transformers Documentation](https://www.sbert.net/)

