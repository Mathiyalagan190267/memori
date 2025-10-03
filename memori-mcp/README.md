# Memori MCP Server

Model Context Protocol (MCP) server for Memori - The Open-Source Memory Layer for AI Agents & Multi-Agent Systems.

## Features

- **MCP Tools** - Three tools for memory operations:
  - `store_memory` - Store new memories with automatic categorization
  - `search_memories` - Semantic search across stored memories
  - `retrieve_memories` - Get recent memories for a user
- **Multi-User Support** - Separate memory spaces per user ID via namespaces
- **Streamable HTTP** - HTTP transport for local MCP server
- **Single Database** - Efficient shared database with namespace isolation
- **FastMCP** - Built on `fastmcp` for easy MCP server development

## What is MCP?

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open protocol that enables AI models to securely access external tools and data sources. This Memori MCP server exposes Memori's memory operations as MCP tools that can be used by any MCP-compatible AI client (like Claude Desktop, IDEs, etc.).

## Installation

**Important**: Install the latest Memori development version first.

1. Install the parent Memori package:
```bash
cd /path/to/memori
pip uninstall -y memorisdk  # Remove old version if installed
pip install -e .
```

2. Install MCP server dependencies:
```bash
cd memori-mcp
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Running the Server

Start the MCP server with HTTP transport:

```bash
python server.py
```

By default, the server runs on `http://0.0.0.0:8080` with streamable HTTP transport.

## Available Tools

### 1. store_memory

Store a new memory for a user.

**Parameters:**
- `user_id` (string, required) - Unique identifier for the user
- `content` (string, required) - The content/text to store
- `metadata` (object, optional) - Additional metadata

**Example:**
```json
{
  "user_id": "alice",
  "content": "User prefers Python and FastAPI for backend development",
  "metadata": {"category": "preference", "priority": "high"}
}
```

**Returns:**
```json
{
  "success": true,
  "memory_id": "uuid-string",
  "message": "Memory stored successfully for user alice"
}
```

### 2. search_memories

Search for memories using semantic search.

**Parameters:**
- `user_id` (string, required) - Unique identifier for the user
- `query` (string, required) - Search query text
- `limit` (integer, optional) - Maximum results (default: 5, max: 100)

**Example:**
```json
{
  "user_id": "alice",
  "query": "What programming languages does the user prefer?",
  "limit": 5
}
```

**Returns:**
```json
{
  "success": true,
  "user_id": "alice",
  "query": "What programming languages...",
  "results": [
    {
      "id": "uuid-string",
      "content": "User prefers Python and FastAPI...",
      "timestamp": "2024-01-15T10:30:00",
      "metadata": {"category": "preference"},
      "similarity_score": 0.95
    }
  ],
  "count": 1
}
```

### 3. retrieve_memories

Retrieve recent memories for a user.

**Parameters:**
- `user_id` (string, required) - Unique identifier for the user
- `limit` (integer, optional) - Maximum memories (default: 10, max: 100)

**Example:**
```json
{
  "user_id": "alice",
  "limit": 10
}
```

**Returns:**
```json
{
  "success": true,
  "user_id": "alice",
  "memories": [
    {
      "id": "uuid-string",
      "content": "User prefers Python...",
      "timestamp": "2024-01-15T10:30:00",
      "metadata": {"category": "preference"}
    }
  ],
  "count": 1
}
```

## Using with MCP Clients

### Claude Desktop

Add to your Claude Desktop MCP configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "memori": {
      "command": "python",
      "args": ["/path/to/memori/memori-mcp/server.py"],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Python MCP Client

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client

async with stdio_client("python", ["server.py"]) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize
        await session.initialize()

        # Store a memory
        result = await session.call_tool(
            "store_memory",
            arguments={
                "user_id": "alice",
                "content": "User loves Python programming",
                "metadata": {"category": "preference"}
            }
        )
        print(result)

        # Search memories
        result = await session.call_tool(
            "search_memories",
            arguments={
                "user_id": "alice",
                "query": "What does the user like?",
                "limit": 5
            }
        )
        print(result)
```

## Architecture

### Single Shared Database
- One Memori instance serves all users
- One database connection/file for efficiency
- Reduces resource usage and improves performance

### Namespace-Based Isolation
- Each user gets a unique namespace: `user_{user_id}`
- Complete data separation between users
- Users cannot access each other's memories

### Streamable HTTP Transport
- HTTP-based MCP protocol
- Supports streaming responses
- Easy integration with web applications
- Can be accessed remotely or locally

## Configuration

Environment variables in `.env`:

```env
# Required: OpenAI API key for memory processing
OPENAI_API_KEY=sk-...

# Optional: Database connection (defaults to SQLite)
DATABASE_CONNECTION=sqlite:///memori_mcp.db

# Optional: Server configuration
MCP_HOST=0.0.0.0
MCP_PORT=8080
```

### Database Options

- **SQLite** (default): `sqlite:///memori_mcp.db`
- **PostgreSQL**: `postgresql://user:password@localhost:5432/memori`
- **MySQL**: `mysql://user:password@localhost:3306/memori`
- **MongoDB**: `mongodb://localhost:27017/memori`

## Development

### Testing the Server

You can test the tools using the FastMCP development UI:

```bash
python server.py
```

Then visit the FastMCP inspector (URL shown in console output).

### Adding New Tools

To add more tools, simply decorate your functions with `@mcp.tool()`:

```python
@mcp.tool()
def your_new_tool(user_id: str, param: str) -> dict:
    """
    Description of your tool.

    Args:
        user_id: User identifier
        param: Your parameter

    Returns:
        Result dictionary
    """
    # Your implementation
    return {"success": True}
```

## Troubleshooting

### "Memori object has no attribute 'add'"

Make sure you've installed the latest Memori development version:
```bash
pip uninstall -y memorisdk
cd /path/to/memori
pip install -e .
```

### Connection Issues

Ensure the server is running and check:
- Firewall settings
- Port availability (default: 8080)
- Environment variables are set correctly

## Resources

- [Memori Documentation](https://github.com/GibsonAI/memori)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)

## License

Same as parent Memori project.
