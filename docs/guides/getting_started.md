# Getting Started

This project is a starting point for a chatbot that uses Retrieval-Augmented Generation (RAG). It has the ability to search across all subject areas or limit searches to selected areas. This will be combined with Role-Based Access Control (RBAC) in a larger project. The system includes conversation history, which is important when working with context-aware responses.

This guide covers the basics of getting the application up and running.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Docker** and **Docker Compose** installed (for Weaviate)
- **Virtual environment** set up (recommended)

## Initial Setup

### 1. Clone and Install Dependencies

First, create a virtual environment and install the required packages:

```bash
# Create virtual environment (if not already created)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Weaviate

Weaviate runs as a Docker container. Start it using Docker Compose:

```bash
# Start Weaviate (runs in the background)
docker-compose up -d

# Verify Weaviate is running
docker-compose ps

# Check Weaviate logs if needed
docker-compose logs weaviate
```

Weaviate will be available at `http://localhost:8080` by default. The gRPC port is `50051`.

### 3. Ingest Data into Weaviate

Before running the application, you need to load data into Weaviate:

```bash
# From the project root directory
python ingest/ingest_weaviate.py ingest/rag_dataset_500.json

# Or use a different data file
python ingest/ingest_weaviate.py ingest/it_help_qna.json
```

This will:

- Create the Weaviate collection schema
- Generate embeddings for your data
- Insert the data into Weaviate

## Configuration

You can configure the application using environment variables. The recommended approach is to use a `.env` file at the project root. You can also set environment variables directly or modify `config.py`.

### Environment Variables

#### Basic Configuration

- `MODEL_TYPE`: Type of model to use (default: `gguf`)
  - Currently supported: `gguf`

#### GGUF Model Configuration

- `MODEL_REPO_ID`: Hugging Face repository ID (default: `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`)
- `MODEL_FILENAME`: Model filename (default: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`)
- `MODEL_N_CTX`: Context window size (default: `4096`)
- `MODEL_N_THREADS`: Number of threads, use `None` for auto-detect (default: `None`)
- `MODEL_N_GPU_LAYERS`: GPU layers for Metal acceleration on M1/M2 Mac (default: `1`)

#### Weaviate Configuration

- `WEAVIATE_HOST`: Weaviate server host (default: `localhost`)
- `WEAVIATE_PORT`: Weaviate HTTP port (default: `8080`)
- `WEAVIATE_GRPC_PORT`: Weaviate gRPC port (default: `50051`)
- `WEAVIATE_COLLECTION_NAME`: Collection name for RAG entries (default: `RAGEntry`)

#### Embedding Model Configuration

- `EMBEDDING_MODEL_NAME`: SentenceTransformer model name (default: `all-MiniLM-L6-v2`)

### Example .env File

Create a `.env` file in the project root:

```env
# Model Configuration
MODEL_TYPE=gguf
MODEL_REPO_ID=TheBloke/Mistral-7B-Instruct-v0.2-GGUF
MODEL_FILENAME=mistral-7b-instruct-v0.2.Q4_K_M.gguf
MODEL_N_CTX=4096
MODEL_N_THREADS=None
MODEL_N_GPU_LAYERS=1

# Weaviate Configuration
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_COLLECTION_NAME=RAGEntry

# Embedding Model
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

## Running the Application

### Using Default Configuration

With a properly configured `.env` file or default settings:

```bash
# Make sure you're in the project root and virtual environment is activated
uvicorn main:app --reload
```

### Using Custom Configuration

You can override settings using environment variables:

```bash
export MODEL_REPO_ID="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
export MODEL_FILENAME="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
export MODEL_N_CTX=4096
export MODEL_N_GPU_LAYERS=1
uvicorn main:app --reload
```

### Using a Different Quantized Model

To use a different quantization level (e.g., higher quality but larger):

```bash
export MODEL_FILENAME="mistral-7b-instruct-v0.2.Q8_0.gguf"
uvicorn main:app --reload
```

### Access the Application

Once the server is running, open your browser to:

```
http://localhost:8000
```

The application provides a web interface for interacting with the chatbot.

## Model Architecture

The application uses an abstract base class pattern for extensibility:

- **`BaseModel`**: Abstract base class defining the model interface
- **`GGUFModel`**: Concrete implementation using `llama-cpp-python` for GGUF models

The models use the singleton pattern to avoid loading the same model multiple times, which saves memory and initialization time.

You can extend the application by creating new model implementations that inherit from `BaseModel`.

## Troubleshooting

### Weaviate Connection Issues

If you encounter connection errors:

1. Verify Weaviate is running: `docker-compose ps`
2. Check Weaviate logs: `docker-compose logs weaviate`
3. Ensure ports are not blocked: `curl http://localhost:8080/v1/meta`

### Model Download Issues

On first run, the model will be downloaded from Hugging Face (approximately 4.5GB). Ensure you have:

- Sufficient disk space
- Stable internet connection
- Proper Hugging Face authentication if required

### Memory Issues

If you experience memory problems:

- Use a lower quantization level (e.g., Q4_K_M instead of Q8_0)
- Reduce `MODEL_N_CTX` (context window size)
- Ensure other applications aren't consuming excessive memory
