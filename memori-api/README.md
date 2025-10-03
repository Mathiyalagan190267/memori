# Memori REST API

FastAPI-based REST API server for Memori - The Open-Source Memory Layer for AI Agents & Multi-Agent Systems.

## Features

- **Record** - Store new memories with automatic categorization
- **Search** - Semantic search across stored memories
- **Retrieve** - Get recent memories for a user
- **Multi-User Support** - Separate memory spaces per user ID
- **Persistent Storage** - All memories stored in database, survive server restarts
- **Configurable** - Support for multiple database backends (SQLite, PostgreSQL, MySQL, MongoDB)

## Installation

**Important**: Make sure to install the latest Memori development version which includes the `add` and `search` methods.

1. Install the parent Memori package (from the root directory):
```bash
cd /path/to/memori
pip uninstall -y memorisdk  # Remove old version if installed
pip install -e .
```

2. Install API dependencies:
```bash
cd memori-api
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

Create a `.env` file with the following variables:

```env
# OpenAI API Key for memory processing
OPENAI_API_KEY=your_openai_api_key_here

# Database Connection String
DATABASE_CONNECTION=sqlite:///memori.db
```

### Database Connection Examples

- **SQLite**: `sqlite:///memori_shared.db` (default - single shared database for all users)
- **PostgreSQL**: `postgresql://user:password@localhost:5432/memori`
- **MySQL**: `mysql://user:password@localhost:3306/memori`
- **MongoDB**: `mongodb://localhost:27017/memori`

### Data Persistence & Multi-User Architecture

**Important**: The API uses a **single shared database** for all users with namespace-based isolation:

✅ **Single Database** - One database file/connection shared by all users
✅ **Namespace Isolation** - Each user gets their own namespace (`user_{user_id}`) for complete data separation
✅ **Memory Efficient** - Single Memori instance serves all users
✅ **Scalable** - Can handle thousands of users in one database
✅ **No Data Loss** - Server restarts don't affect stored memories
✅ **Production-ready** - Efficient architecture for real-world deployments

## API Endpoints

### 1. Record New Data

**POST** `/api/v1/record`

Store new information in memory.

**Request Body:**
```json
{
  "user_id": "user_001",
  "content": "User prefers Python and FastAPI for backend development",
  "metadata": {
    "category": "preference",
    "priority": "high"
  }
}
```

**Response:**
```json
{
  "success": true,
  "memory_id": "uuid-string",
  "message": "Memory recorded successfully"
}
```

### 2. Search for Data

**POST** `/api/v1/search`

Perform semantic search across memories.

**Request Body:**
```json
{
  "user_id": "user_001",
  "query": "What programming languages does the user prefer?",
  "limit": 5
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": "uuid-string",
      "user_id": "user_001",
      "content": "User prefers Python and FastAPI...",
      "timestamp": "2024-01-15T10:30:00",
      "metadata": {"category": "preference"},
      "similarity_score": 0.95
    }
  ],
  "count": 1
}
```

### 3. Retrieve Data

**POST** `/api/v1/retrieve`

Retrieve recent memories for a user.

**Request Body:**
```json
{
  "user_id": "user_001",
  "limit": 10
}
```

**Response:**
```json
{
  "success": true,
  "memories": [
    {
      "id": "uuid-string",
      "user_id": "user_001",
      "content": "User prefers Python...",
      "timestamp": "2024-01-15T10:30:00",
      "metadata": {"category": "preference"}
    }
  ],
  "count": 1
}
```

### Health Check

**GET** `/health`

Check API server status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "persistence": "database"
}
```

## Example Usage

### Using cURL

```bash
# Record a memory
curl -X POST "http://localhost:8000/api/v1/record" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "content": "User prefers clean, well-documented code with type hints",
    "metadata": {"category": "preference"}
  }'

# Search memories
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "query": "What are the user coding preferences?",
    "limit": 5
  }'

# Retrieve recent memories
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "limit": 10
  }'
```

### Using Python

```python
import requests

API_URL = "http://localhost:8000"

# Record a memory
response = requests.post(
    f"{API_URL}/api/v1/record",
    json={
        "user_id": "user_001",
        "content": "User is working on a FastAPI project",
        "metadata": {"category": "project"}
    }
)
print(response.json())

# Search memories
response = requests.post(
    f"{API_URL}/api/v1/search",
    json={
        "user_id": "user_001",
        "query": "What project is the user working on?",
        "limit": 5
    }
)
print(response.json())

# Retrieve memories
response = requests.post(
    f"{API_URL}/api/v1/retrieve",
    json={
        "user_id": "user_001",
        "limit": 10
    }
)
print(response.json())
```

## Interactive API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Architecture

The API server uses a **shared database with namespace isolation** approach:

### Single Shared Database
- One Memori instance serves all users
- One database connection/file for the entire application
- Reduces resource usage and improves performance

### Namespace-Based Isolation
- Each user automatically gets a unique namespace: `user_{user_id}`
- Namespaces ensure complete data separation between users
- Users cannot access each other's memories

### Key Features
- **Database Persistence**: All memories stored in SQLite/PostgreSQL/MySQL/MongoDB
- **Memory Efficient**: Single instance instead of one per user
- **Scalable**: Handles thousands of users efficiently
- **Thread-Safe**: Namespace is set per-request
- **No Data Loss**: Survives server restarts, crashes, and deployments
- **Intelligent Processing**: Conscious ingestion for AI-powered categorization

## License

Same as parent Memori project.