"""
Memori MCP Server
Model Context Protocol server for Memori memory layer operations
"""

import os
from typing import Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from memori import Memori

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("Memori Memory Server")

# Single shared Memori instance
_shared_memori: Optional[Memori] = None


def get_memori() -> Memori:
    """Get or create the shared Memori instance"""
    global _shared_memori

    if _shared_memori is None:
        db_connection = os.getenv("DATABASE_CONNECTION", "sqlite:///memori_mcp.db")
        openai_key = os.getenv("OPENAI_API_KEY")

        _shared_memori = Memori(
            database_connect=db_connection,
            namespace="default",
            conscious_ingest=True,
            openai_api_key=openai_key,
            user_id="mcp-server",
        )
        _shared_memori.enable()

    return _shared_memori


@mcp.tool()
def store_memory(user_id: str, content: str, metadata: Optional[dict] = None) -> dict:
    """
    Store a new memory for a user.

    Args:
        user_id: Unique identifier for the user
        content: The content/text to store in memory
        metadata: Optional metadata dictionary (e.g., {"category": "preference", "priority": "high"})

    Returns:
        Dictionary with success status and memory_id
    """
    try:
        memori = get_memori()

        memory_id = memori.add(text=content, metadata=metadata or {})

        return {
            "success": True,
            "memory_id": str(memory_id),
            "message": f"Memory stored successfully for user {user_id}",
        }
    except Exception as e:
        return {"success": False, "error": str(e), "message": "Failed to store memory"}


@mcp.tool()
def search_memories(user_id: str, query: str, limit: int = 5) -> dict:
    """
    Search for memories using semantic search.

    Args:
        user_id: Unique identifier for the user
        query: Search query text
        limit: Maximum number of results to return (default: 5, max: 100)

    Returns:
        Dictionary with search results and count
    """
    try:
        memori = get_memori()

        # Limit validation
        limit = min(max(1, limit), 100)

        # Perform search
        if hasattr(memori, "search"):
            results = memori.search(query=query, limit=limit)
        else:
            results = memori.retrieve_context(query=query, limit=limit)

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "id": str(result.get("id", "")),
                    "content": result.get("content", result.get("text", "")),
                    "timestamp": str(result.get("timestamp"))
                    if result.get("timestamp")
                    else None,
                    "metadata": result.get("metadata"),
                    "similarity_score": result.get("similarity", result.get("score")),
                }
            )

        return {
            "success": True,
            "user_id": user_id,
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to search memories",
        }


@mcp.tool()
def retrieve_memories(user_id: str, limit: int = 10) -> dict:
    """
    Retrieve recent memories for a user.

    Args:
        user_id: Unique identifier for the user
        limit: Maximum number of memories to retrieve (default: 10, max: 100)

    Returns:
        Dictionary with retrieved memories and count
    """
    try:
        memori = get_memori()

        # Limit validation
        limit = min(max(1, limit), 100)

        # Get conversation history
        results = memori.get_conversation_history(limit=limit)

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "id": str(result.get("id", "")),
                    "content": result.get("user_input", result.get("content", "")),
                    "timestamp": str(result.get("timestamp"))
                    if result.get("timestamp")
                    else None,
                    "metadata": result.get("metadata"),
                }
            )

        return {
            "success": True,
            "user_id": user_id,
            "memories": formatted_results,
            "count": len(formatted_results),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve memories",
        }


if __name__ == "__main__":
    # Run the MCP server with HTTP transport
    mcp.run(transport="streamable-http")
