"""Main application file for the Mistral Chatbot."""

import sqlite3
import secrets
import traceback
import time
from html import escape
from typing import Optional, List
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from models import BaseModel, GGUFModel
from rag import WeaviateRAG
from config import get_settings

settings = get_settings()

app = FastAPI(title="Mistral Chatbot")
templates = Jinja2Templates(directory="templates")


def create_model_instance() -> BaseModel:
    """
    Create a model instance based on configuration from settings.

    Returns:
        Model instance implementing BaseModel
    """
    if settings.model_type.lower() == "gguf":
        model_config = settings.get_model_config_dict()
        return GGUFModel(config=model_config)
    else:
        raise ValueError(f"Unsupported model type: {settings.model_type}")


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on application startup."""
    print("Initializing RAG system...")
    print(f"Model configuration: {settings.get_model_config_dict()}")
    print(f"Weaviate: {settings.weaviate_host}:{settings.weaviate_port}")

    # Initialize RAG system with model configuration
    rag = WeaviateRAG(
        model_class=GGUFModel,
        model_config=None,  # Will use settings.get_model_config_dict()
    )

    # Load model through RAG (uses singleton pattern)
    model = rag.get_or_create_model()
    model.load()

    # Warmup: Run a dummy query to initialize SentenceTransformer and LLM
    # This eliminates first-request delays
    print("Warming up models (this may take a moment)...")
    try:
        # Warmup embedding model with a dummy encode
        _ = rag.embedding_model.encode("warmup")
        print("Embedding model warmed up")

        # Warmup LLM with a short generation
        _ = model.generate("warmup", max_tokens=5, temperature=0.7)
        print("LLM model warmed up")
    except RuntimeError as e:
        print(f"Warning: Warmup failed (non-critical): {e}")
    except ValueError as e:
        print(f"Warning: Warmup failed (non-critical): {e}")
    except AttributeError as e:
        print(f"Warning: Warmup failed (non-critical): {e}")

    # Store RAG instance in app state instead of global variable
    app.state.rag = rag
    print("RAG system initialized successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    print("Shutting down application...")
    rag = getattr(app.state, "rag", None)
    if rag:
        # Get model and unload it
        try:
            model = rag.get_or_create_model()
            if model and hasattr(model, "is_loaded") and model.is_loaded():
                print("Unloading model...")
                model.unload()
        except (ValueError, AttributeError, RuntimeError) as e:
            print(f"Error unloading model: {e}")

        # Close RAG connections
        try:
            print("Closing RAG connections...")
            rag.close()
        except (AttributeError, RuntimeError, ConnectionError) as e:
            print(f"Error closing RAG connections: {e}")
        finally:
            app.state.rag = None
    print("Application shutdown complete")


def get_db_connection():
    """Get SQLite database connection."""
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


def create_session() -> str:
    """Generate a new session ID."""
    return secrets.token_hex(16)


def get_history_html(session_id: str) -> str:
    """Get chat history as HTML for a session."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT user_message, bot_response FROM chat_history WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()

    history_html = ""
    for row in rows:
        history_html += f'<div class="p-2 bg-gray-200 rounded mb-2"><strong>You:</strong> {row["user_message"]}</div>'
        history_html += f'<div class="p-2 bg-blue-100 rounded mb-2"><strong>Bot:</strong> {row["bot_response"]}</div>'

    return history_html


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve the main chat interface."""
    # Initialize database on first request
    init_db()

    # Get or create session ID
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = create_session()

    # Get chat history
    history_html = get_history_html(session_id)

    response = templates.TemplateResponse("index.html", {"request": request, "history_html": history_html})

    # Set session cookie if it doesn't exist
    if not request.cookies.get("session_id"):
        response.set_cookie(key="session_id", value=session_id)

    return response


@app.post("/chat", response_class=HTMLResponse)
async def chat_endpoint(
    request: Request,
    prompt: str = Form(...),
    sources: Optional[List[str]] = Form(None),
):
    """Handle chat requests using RAG (Retrieval-Augmented Generation)."""
    # Get RAG instance from app state (set during startup)
    rag: Optional[WeaviateRAG] = getattr(app.state, "rag", None)

    if not rag:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    try:
        session_id = request.cookies.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session not found")

        # Build filters if sources are selected
        filters = None
        if sources:
            # Ensure sources is a list (FastAPI might send a single string or list)
            sources_list = sources if isinstance(sources, list) else [sources]
            if len(sources_list) > 0:
                filters = {"sources": sources_list}

        # Start timing the request
        request_start = time.time()

        # Generate bot response using RAG (retrieves context from Weaviate)
        rag_start = time.time()
        rag_result = rag.generate_with_context(
            query=prompt,
            model=None,  # Uses model from RAG instance
            top_k=settings.rag_top_k,
            max_tokens=settings.rag_max_tokens,
            temperature=settings.rag_temperature,
            filters=filters,
        )
        rag_time = time.time() - rag_start

        response_text = rag_result.get("response", "")

        # Extract timing and context info from RAG result
        context_size = rag_result.get("context_size", 0)
        context_chars = rag_result.get("context_chars", 0)
        retrieval_time = rag_result.get("retrieval_time", 0)
        embedding_time = rag_result.get("embedding_time", 0)
        generation_time = rag_result.get("generation_time", 0)

        # Ensure we have a response
        if not response_text or not response_text.strip():
            response_text = "I'm sorry, I couldn't generate a response. Please try again."

        # Use try/finally to ensure connection is always closed
        db_start = time.time()
        conn = None
        try:
            conn = get_db_connection()
            # Insert into database
            conn.execute(
                "INSERT INTO chat_history (session_id, user_message, bot_response) VALUES (?, ?, ?)",
                (session_id, prompt, response_text),
            )
            conn.commit()
        finally:
            if conn:
                conn.close()
        db_time = time.time() - db_start

        # Log timing information (after all operations complete)
        total_time = time.time() - request_start
        print(f"\n{'='*70}")
        print(f"REQUEST TIMING - Query: '{prompt[:60]}...'")
        print(f"{'='*70}")
        print(f"  Total Request Time:     {total_time:.3f}s")
        print(f"  ┌─ RAG Pipeline:         {rag_time:.3f}s")
        print(f"  │  ├─ Embedding:        {embedding_time:.3f}s")
        print(f"  │  ├─ Retrieval:        {retrieval_time:.3f}s")
        print(f"  │  └─ Generation:       {generation_time:.3f}s")
        print(f"  └─ Database Write:      {db_time:.3f}s")
        print(f"  Context Size:           {context_size} tokens ({context_chars:,} chars)")
        print(
            f"  Prompt Size:            {rag_result.get('prompt_tokens', 0)} tokens ({rag_result.get('prompt_chars', 0):,} chars)"
        )
        print(f"  Response Length:        {len(response_text):,} chars")
        print(f"  Configuration:          top_k={settings.rag_top_k}, max_tokens={settings.rag_max_tokens}")
        print(f"{'='*70}\n")

        # Escape HTML in user input and response to prevent XSS
        escaped_prompt = escape(prompt)
        escaped_response = escape(response_text)

        # Return HTML response for HTMX
        html_response = (
            f'<div class="p-2 bg-gray-200 rounded mb-2"><strong>You:</strong> {escaped_prompt}</div>'
            f'<div class="p-2 bg-blue-100 rounded mb-2"><strong>Bot:</strong> {escaped_response}</div>'
        )

        return HTMLResponse(content=html_response)
    except HTTPException as e:
        # Return HTTP exceptions as HTML for HTMX
        error_html = (
            f'<div class="p-2 bg-red-100 rounded mb-2 border border-red-300">'
            f"<strong>Error:</strong> {e.detail}"
            f"</div>"
        )
        return HTMLResponse(content=error_html, status_code=e.status_code)
    except (ValueError, RuntimeError, ConnectionError, AttributeError):
        # Log the full traceback for debugging
        error_traceback = traceback.format_exc()
        print(f"Error in chat_endpoint: {error_traceback}")

        # Return error as HTML for HTMX
        error_html = (
            '<div class="p-2 bg-red-100 rounded mb-2 border border-red-300">'
            "<strong>Error:</strong> An error occurred while processing your request. Please try again."
            "</div>"
        )
        return HTMLResponse(content=error_html, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=settings.fastapi_reload,
    )
