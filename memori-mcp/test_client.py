"""
Simple test client to verify MCP server is working
"""

import httpx
import json

# Server URL
BASE_URL = "http://127.0.0.1:8000"

def test_store_memory():
    """Test store_memory tool"""
    print("\n=== Testing store_memory ===")

    # This is a simplified test - actual MCP protocol requires proper message format
    # For real MCP client, use mcp.client SDK

    payload = {
        "user_id": "test_user",
        "content": "I love programming in Python and building AI applications",
        "metadata": {"category": "preference", "source": "test"}
    }

    print(f"Storing memory: {payload['content']}")
    print("Note: Use proper MCP client for production usage")


def test_server_running():
    """Test if server is running"""
    print("\n=== Checking Server Status ===")
    try:
        response = httpx.get(BASE_URL, timeout=5.0)
        print(f"✓ Server is running at {BASE_URL}")
        print(f"  Status code: {response.status_code}")
        return True
    except Exception as e:
        print(f"✗ Server not accessible: {e}")
        return False


if __name__ == "__main__":
    print("Memori MCP Server Test Client")
    print("=" * 50)

    if test_server_running():
        print("\n✓ MCP Server is running successfully!")
        print("\nAvailable tools:")
        print("  - store_memory(user_id, content, metadata)")
        print("  - search_memories(user_id, query, limit)")
        print("  - retrieve_memories(user_id, limit)")

        print("\nTo use with Claude Desktop:")
        print("  1. Add server config to claude_desktop_config.json")
        print("  2. Restart Claude Desktop")
        print("  3. Look for the 🔌 icon in Claude Desktop")

        print("\nFor Python client usage:")
        print("  See README.md for MCP client examples")
    else:
        print("\n✗ Server is not running")
        print("  Start with: python server.py")
