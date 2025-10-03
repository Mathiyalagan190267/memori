"""
Memori REST API Server
FastAPI implementation for Memori memory layer operations
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn
import os
from dotenv import load_dotenv

from memori import Memori

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Memori API",
    description="REST API for Memori - The Open-Source Memory Layer for AI Agents",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared Memori instance with one database
# User isolation is handled via namespace parameter
_shared_memori: Optional[Memori] = None


# Pydantic models for request/response
class RecordRequest(BaseModel):
    """Request model for recording new data"""

    user_id: str = Field(..., description="Unique identifier for the user")
    content: str = Field(..., description="Content to be stored in memory")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class SearchRequest(BaseModel):
    """Request model for searching data"""

    user_id: str = Field(..., description="Unique identifier for the user")
    query: str = Field(..., description="Search query text")
    limit: Optional[int] = Field(
        default=5, gt=0, le=100, description="Maximum number of results"
    )


class RetrieveRequest(BaseModel):
    """Request model for retrieving data"""

    user_id: str = Field(..., description="Unique identifier for the user")
    limit: Optional[int] = Field(
        default=10, gt=0, le=100, description="Maximum number of memories to retrieve"
    )


class MemoryResponse(BaseModel):
    """Response model for memory data"""

    id: str
    user_id: str
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    similarity_score: Optional[float] = None


class RecordResponse(BaseModel):
    """Response model for record operation"""

    success: bool
    memory_id: str
    message: str


class SearchResponse(BaseModel):
    """Response model for search operation"""

    success: bool
    results: List[MemoryResponse]
    count: int


class RetrieveResponse(BaseModel):
    """Response model for retrieve operation"""

    success: bool
    memories: List[MemoryResponse]
    count: int


def get_shared_memori() -> Memori:
    """
    Get or create the shared Memori instance.

    Uses a single database for all users with namespace-based isolation.
    Each user_id becomes a namespace to keep their memories separate.
    """
    global _shared_memori

    if _shared_memori is None:
        # Get configuration from environment variables
        db_connection = os.getenv("DATABASE_CONNECTION", "sqlite:///memori_shared.db")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Create single shared instance
        _shared_memori = Memori(
            database_connect=db_connection,
            namespace="default",  # Default namespace, will be overridden per-request
            conscious_ingest=True,
            openai_api_key=openai_key,
            user_id="api-server"
        )
        _shared_memori.enable()

    return _shared_memori


def set_user_context(memori: Memori, user_id: str):
    """
    Set the user context (namespace) for the current operation.
    Each user gets their own namespace for complete data isolation.
    """
    memori.namespace = f"user_{user_id}"
    memori.user_id = user_id


@app.on_event("startup")
async def startup_event():
    """Initialize API server on startup"""
    # Pre-initialize the shared Memori instance
    try:
        memori = get_shared_memori()
        print("✓ Memori API server initialized successfully")
        print(f"  Database: {os.getenv('DATABASE_CONNECTION', 'sqlite:///memori_shared.db')}")
        print(f"  Memori instance created: {memori is not None}")
        print(f"  Has 'add' method: {hasattr(memori, 'add')}")
        print("  Endpoints available:")
        print("    - POST /api/v1/record")
        print("    - POST /api/v1/search")
        print("    - POST /api/v1/retrieve")
        print("  Data persistence: All memories stored in database")
    except Exception as e:
        print(f"✗ Failed to initialize Memori: {e}")
        print("  API will attempt to initialize on first request")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("✓ Memori API server shutdown complete")
    print("  All user memories persisted in database")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Memori API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "record": "/api/v1/record",
            "search": "/api/v1/search",
            "retrieve": "/api/v1/retrieve",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "persistence": "database",
    }


@app.post(
    "/api/v1/record", response_model=RecordResponse, status_code=status.HTTP_201_CREATED
)
async def record_memory(request: RecordRequest):
    """
    Record new data into memory

    Store new information associated with a specific user ID.
    The content will be processed and stored with automatic categorization.

    Example:
        ```json
        {
            "user_id": "user_001",
            "content": "User prefers Python and FastAPI for backend development",
            "metadata": {"category": "preference", "priority": "high"}
        }
        ```
    """
    try:
        # Get shared Memori instance and set user context
        memori = get_shared_memori()
        set_user_context(memori, request.user_id)

        # Add the memory using Memori's add method
        memory_id = memori.add(text=request.content, metadata=request.metadata)

        return RecordResponse(
            success=True,
            memory_id=str(memory_id),
            message="Memory recorded successfully",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record memory: {str(e)}",
        )


@app.post("/api/v1/search", response_model=SearchResponse)
async def search_memory(request: SearchRequest):
    """
    Search for data in memory

    Perform semantic search across stored memories for a specific user.
    Returns relevant memories ranked by similarity to the query.

    Example:
        ```json
        {
            "user_id": "user_001",
            "query": "What programming languages does the user prefer?",
            "limit": 5
        }
        ```
    """
    try:
        # Get shared Memori instance and set user context
        memori = get_shared_memori()
        set_user_context(memori, request.user_id)

        # Perform search using Memori's retrieve_context method
        # (search() is an alias for retrieve_context in newer versions)
        if hasattr(memori, 'search'):
            results = memori.search(query=request.query, limit=request.limit)
        else:
            results = memori.retrieve_context(query=request.query, limit=request.limit)

        # Format results
        memories = []
        for result in results:
            memory = MemoryResponse(
                id=str(result.get("id", "")),
                user_id=request.user_id,
                content=result.get("content", result.get("text", "")),
                timestamp=str(result.get("timestamp"))
                if result.get("timestamp")
                else None,
                metadata=result.get("metadata"),
                similarity_score=result.get("similarity", result.get("score")),
            )
            memories.append(memory)

        return SearchResponse(success=True, results=memories, count=len(memories))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search memory: {str(e)}",
        )


@app.post("/api/v1/retrieve", response_model=RetrieveResponse)
async def retrieve_memory(request: RetrieveRequest):
    """
    Retrieve data from memory

    Retrieve recent memories for a given user.
    Returns memories ordered by recency.

    Example:
        ```json
        {
            "user_id": "user_001",
            "limit": 10
        }
        ```
    """
    try:
        # Get shared Memori instance and set user context
        memori = get_shared_memori()
        set_user_context(memori, request.user_id)

        # Get conversation history (which includes all recorded memories)
        results = memori.get_conversation_history(limit=request.limit)

        # Format results
        memories = []
        for result in results:
            memory = MemoryResponse(
                id=str(result.get("id", "")),
                user_id=request.user_id,
                content=result.get("user_input", result.get("content", "")),
                timestamp=str(result.get("timestamp"))
                if result.get("timestamp")
                else None,
                metadata=result.get("metadata"),
            )
            memories.append(memory)

        return RetrieveResponse(success=True, memories=memories, count=len(memories))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
