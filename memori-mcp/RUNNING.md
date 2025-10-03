# Memori MCP Server - Running Status

## ✓ Server is Running

The MCP server is currently running on:
- **URL**: `http://127.0.0.1:8000`
- **Transport**: Streamable HTTP
- **Status**: Active

## Virtual Environment

Location: `memori-mcp/venv/`

To activate:
```bash
source venv/bin/activate
```

## Server Management

### Start Server
```bash
source venv/bin/activate
python server.py
```

### Stop Server
Press `CTRL+C` or:
```bash
pkill -f "python server.py"
```

### Check Status
```bash
python test_client.py
```

## Available MCP Tools

1. **store_memory**
   - Stores new memories for a user
   - Parameters: user_id, content, metadata

2. **search_memories**
   - Semantic search across memories
   - Parameters: user_id, query, limit

3. **retrieve_memories**
   - Get recent memories
   - Parameters: user_id, limit

## Next Steps

### Use with Claude Desktop

1. Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "memori": {
      "command": "/Users/boburumurzokov/Gibson/memori/memori-mcp/venv/bin/python",
      "args": ["/Users/boburumurzokov/Gibson/memori/memori-mcp/server.py"],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

2. Restart Claude Desktop

3. Look for the 🔌 icon to see available tools

### Use with Python MCP Client

See `README.md` for Python client examples.

## Configuration

Edit `.env` file to configure:
- `OPENAI_API_KEY` - Required for memory processing
- `DATABASE_CONNECTION` - Database location (default: sqlite:///memori_mcp.db)
- `MCP_HOST` - Server host (default: 0.0.0.0)
- `MCP_PORT` - Server port (default: 8080)

## Logs

Server logs are displayed in the terminal where the server is running.

## Troubleshooting

**Server not starting:**
- Check if port 8000 is already in use
- Ensure virtual environment is activated
- Verify Memori package is installed: `pip show memorisdk`

**"Memori object has no attribute 'add'":**
- Make sure latest Memori is installed: `pip install -e /path/to/memori`

**OpenAI API errors:**
- Set `OPENAI_API_KEY` in `.env` file
- Verify the API key is valid

## Database

Location: `memori_mcp.db` (SQLite)

Each user gets their own namespace for data isolation:
- User "alice" → namespace "user_alice"
- User "bob" → namespace "user_bob"

All data persists across server restarts.
