import weaviate
import json
import argparse
import sys
from pathlib import Path
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

# Parse command line arguments
parser = argparse.ArgumentParser(description="Ingest data into Weaviate")
parser.add_argument("data_file", type=str, help="Path to the JSON data file to ingest")
args = parser.parse_args()

# Load configuration from environment
settings = get_settings()

# ---- CONFIG ----
CLASS_NAME = settings.weaviate_collection_name
DATA_FILE = args.data_file
EMBEDDING_MODEL = settings.embedding_model_name

# ---- CONNECT TO WEAVIATE (v4 API) ----
# Option 1: With gRPC (requires port 50051 exposed in docker-compose.yaml)
# Option 2: HTTP-only mode (works without gRPC, slower but functional)
client = weaviate.connect_to_local(
    host=settings.weaviate_host,
    port=settings.weaviate_port,
    grpc_port=settings.weaviate_grpc_port,
    skip_init_checks=True,  # Skip gRPC health check to use HTTP-only mode if needed
)

try:
    # ---- DEFINE SCHEMA (v4 API) ----
    if client.collections.exists(CLASS_NAME):
        print(f"Collection {CLASS_NAME} already exists. Skipping schema creation.")
    else:
        # In v4, use Configure.Vectorizer.none() to disable vectorizer and provide own embeddings
        client.collections.create(
            name=CLASS_NAME,
            description="RAG question-answer pairs for chatbot testing",
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # We'll provide our own embeddings
            properties=[
                wvc.config.Property(
                    name="question", data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(name="answer", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="context", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
            ],
        )
        print(f"Created collection: {CLASS_NAME}")

    # ---- LOAD DATA ----
    # Handle relative paths - if file doesn't exist, try relative to script directory
    data_path = Path(DATA_FILE)
    if not data_path.is_absolute() and not data_path.exists():
        # Try relative to script directory (ingest/)
        data_path = Path(__file__).parent / DATA_FILE
    if not data_path.exists():
        # Try relative to project root
        data_path = Path(__file__).parent.parent / DATA_FILE

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE} (tried: {data_path})"
        )

    print(f"Loading data from {data_path}...")

    with open(data_path, "r") as f:
        entries = json.load(f)

    # ---- EMBEDDING MODEL ----
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ---- BATCH INSERT (v4 API) ----
    collection = client.collections.get(CLASS_NAME)

    print(f"Inserting {len(entries)} entries...")
    with collection.batch.dynamic() as batch:
        for i, entry in enumerate(entries):
            # Generate embedding for the question
            vector = model.encode(entry["question"]).tolist()

            # Insert data object with vector
            batch.add_object(
                properties={
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "context": entry.get("context", ""),
                    "source": entry.get("source", ""),
                },
                vector=vector,
            )

            if (i + 1) % 50 == 0:
                print(f"Inserted {i + 1}/{len(entries)} entries...")

    print(f"Successfully inserted {len(entries)} entries into {CLASS_NAME}!")

finally:
    # Always close the connection
    client.close()
