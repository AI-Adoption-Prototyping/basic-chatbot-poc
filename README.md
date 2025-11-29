# Basic RAG Chatbot

A proof-of-concept chatbot demonstrating Retrieval-Augmented Generation (RAG) running efficiently on modest hardware. This project showcases how to build an in-house AI solution using open-source tools, with support for role-based access control example via filtering, session history, and flexible querying.

## Overview

This project started as a simple proof of concept (`load_model.py`) and evolved into a functional prototype during development sessions. The goal was to demonstrate that in-house AI is practical for individuals or small teams, with approximately **30 hours** of development time (about 20 hours coding, 10 hours documentation and setup).

### Key Features

- **Retrieval-Augmented Generation (RAG)**: Context-aware responses using vector search
- **Session History**: Maintains conversation context across interactions
- **Source Filtering**: Filter responses by data source or category
- **Role-Based Access Control (RBAC)**: Foundation for limiting data access by Filtering
- **Multiple Search Strategies**: Vector, keyword, and hybrid search support
- **Highly Configurable**: Environment-based configuration for all components
- **Comprehensive Documentation**: Detailed guides covering all aspects of the system

## Architecture

### Core Components

- **LLM**: Mistral-7B-Instruct (GGUF quantized) via llama.cpp
- **Vector Database**: Weaviate for semantic search and filtering
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Web Framework**: FastAPI with HTMX for interactive UI
- **Storage**: SQLite for chat history

### Design Principles

- **Maximize Autonomy and Flexibility**
  - Work directly with core technologies
  - Avoid pre-packaged RAG environments
  - Open-source licensing throughout

- **Forward Thinking**
  - **Data Access**: Filtering capabilities for role-based access, GraphQL-style queries, direct search
  - **Base Models**: Swappable LLM and RAG implementations via abstract base classes

- **Base Features**
  - Session history management
  - Filtering examples and demonstrations
  - Strong documentation (see `docs/` directory)
  - Highly configurable via environment variables
  - Example scripts and helper utilities

## Quick Start

### Prerequisites

- **Python 3.8+**
- **Docker** and **Docker Compose** (for Weaviate)
- **Virtual environment** (recommended)

### Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Weaviate**:

   ```bash
   docker-compose up -d
   ```

5. **Ingest sample data**:

   ```bash
   python ingest/ingest_weaviate.py ingest/rag_dataset_500.json
   ```

6. **Run the application**:

   ```bash
   uvicorn main:app --reload
   ```

7. **Open your browser** to `http://localhost:8000`

### Configuration

Configuration is managed through environment variables. Create a `.env` file in the project root:

```env
# Model Configuration
MODEL_TYPE=gguf
MODEL_REPO_ID=TheBloke/Mistral-7B-Instruct-v0.2-GGUF
MODEL_FILENAME=mistral-7b-instruct-v0.2.Q4_K_M.gguf
MODEL_N_CTX=4096
MODEL_N_GPU_LAYERS=1

# Weaviate Configuration
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_COLLECTION_NAME=RAGEntry

# Embedding Model
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

See `ENV_EXAMPLE` for all available configuration options.

## Project Structure

```
.
├── docs/                    # Documentation (MkDocs)
│   ├── guides/             # Detailed guides
│   └── index.md            # Documentation homepage
├── ingest/                 # Data ingestion scripts
│   ├── ingest_weaviate.py  # Load data into Weaviate
│   └── *.json             # Sample data files
├── models/                  # LLM model implementations
│   ├── base.py             # BaseModel abstract class
│   └── gguf_model.py       # GGUF model implementation
├── rag/                    # RAG implementations
│   ├── base.py             # BaseRAG abstract class
│   └── weaviate_rag.py     # Weaviate RAG implementation
├── templates/               # HTML templates
│   └── index.html          # Chat interface
├── main.py                  # FastAPI application
├── config.py                # Configuration management
├── query_cli.py             # CLI for querying Weaviate
├── query_weaviate.py        # Query examples
├── rag_example.py           # RAG usage examples
└── load_model.py            # Model loading example
```

## Helper Scripts

The project includes several helper scripts for development and testing:

- **`ingest/ingest_weaviate.py`**: Load JSON data into Weaviate with embeddings
- **`query_cli.py`**: Command-line interface for querying Weaviate
- **`query_weaviate.py`**: Examples of different query strategies
- **`rag_example.py`**: Complete RAG pipeline examples

See the [Helper Scripts documentation](docs/guides/scripts.md) for detailed usage.

## Documentation

Comprehensive documentation is available in the `docs/` directory. To view it:

```bash
pip install mkdocs-material
mkdocs serve
```

Then open `http://localhost:8000` in your browser.

### Documentation Topics

- **[Getting Started](docs/guides/getting_started.md)**: Setup and configuration guide
- **[Overview](docs/guides/overview.md)**: Architecture and design decisions
- **[LLM Model Selection](docs/guides/llm_model_selection.md)**: Model comparison and selection
- **[Vector Database](docs/guides/vector_database.md)**: Weaviate setup and usage
- **[Sentence Embedding Model](docs/guides/sentence_embedding_model.md)**: Embedding model details
- **[Quantized LLM](docs/guides/quantized_large_language_model.md)**: GGUF format and quantization
- **[RAG](docs/guides/rag.md)**: Retrieval-Augmented Generation implementation
- **[Querying Weaviate](docs/guides/query.md)**: Query strategies and examples
- **[Helper Scripts](docs/guides/scripts.md)**: Script documentation and usage
- **[Optimization](docs/guides/generation_optimization.md)**: Optimizing initial configuration M1-Mac

## Usage Examples

### Basic Chat

Start the application and interact through the web interface at `http://localhost:8000`.

### Query Weaviate via CLI

```bash
python query_cli.py --query "machine learning" --limit 5
python query_cli.py --query "python" --method hybrid --source "Academic Fields"
```

### Run RAG Examples

```bash
python rag_example.py
```

## Technology Stack

- **FastAPI**: Modern Python web framework
- **llama.cpp**: Efficient CPU-based LLM inference
- **Weaviate**: Vector database with filtering capabilities
- **SentenceTransformers**: Semantic embeddings
- **HTMX**: Dynamic web interactions
- **SQLite**: Chat history storage

## Development Notes

This is a proof-of-concept project. While the core components are implemented properly, not all production best practices were followed. The focus was on:

- Demonstrating RAG on modest hardware
- Proving in-house AI feasibility
- Learning through hands-on implementation
- Building a foundation for future enhancements

## Goals Achieved

✅ Efficient RAG on CPU-only hardware  
✅ Session history management  
✅ Source filtering capabilities  
✅ Role-based access control foundation  
✅ Swappable model and RAG implementations  
✅ Comprehensive documentation  
✅ Example scripts and helpers  

## Future Enhancements

- Enhanced role-based access control
- Multi-tenant support
- Advanced chunking strategies
- Fine-tuned embedding models
- Additional LLM model support
- Production deployment guides

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project uses open-source components with various licenses. See individual component licenses for details:
- **Mistral-7B-Instruct**: Apache 2.0 License
- **Weaviate**: BSD 3-Clause License
- **llama.cpp**: MIT License
- **Sentence Transformers**: Apache 2.0 License

## Contributing

This is a personal learning project, but suggestions and improvements are welcome. Please review the documentation before contributing.

## Acknowledgments

- **TheBloke** for quantized GGUF models
- **Weaviate** team for the excellent vector database
- **Sentence Transformers** for embedding models
- **llama.cpp** for efficient CPU inference

---

For detailed information, see the [documentation](docs/guides/getting_started.md).
