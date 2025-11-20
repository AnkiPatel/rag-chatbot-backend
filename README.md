# RAG Chatbot Backend

A production-ready RAG (Retrieval-Augmented Generation) chatbot backend that processes PDF documents and integrates web search capabilities. Built with FastAPI, OpenAI, ChromaDB, and Tavily Search.

## Features

- 📚 **PDF Knowledge Base**: Process and index product/admin guide PDFs
- 🔍 **Semantic Search**: Vector-based document retrieval using ChromaDB
- 🌐 **Web Search Integration**: Supplement answers with real-time web search (Tavily API)
- 🤖 **LLM-Powered Responses**: Generate contextual answers using OpenAI GPT
- 🚀 **REST API**: Clean API endpoints for easy integration
- 📊 **Confidence Scoring**: Transparent relevance metrics for answers
- ⚡ **Streaming Support**: Real-time response streaming
- 🐳 **Docker Ready**: Containerized deployment with Docker Compose

## Architecture

```
User Query → Vector Search (ChromaDB) → Web Search (if needed) → LLM (OpenAI) → Response
                     ↓
              PDF Knowledge Base
```

## Prerequisites

- Python 3.11+
- OpenAI API key
- Tavily API key

## Quick Start

### 1. Clone and Setup

```bash
cd rag-chatbot-backend
```

### 2. Create Environment File

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add:
- `OPENAI_API_KEY=your-openai-api-key`
- `TAVILY_API_KEY=your-tavily-api-key`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add PDF Documents

Place your PDF files (product guides, admin guides) in:
```
data/pdfs/
```

### 5. Index Documents

Run the indexing script:

```bash
python -c "from app.services.pdf_processor import PDFProcessor; from app.services.vector_store import VectorStore; processor = PDFProcessor(); store = VectorStore(); chunks = processor.process_directory(); store.add_documents(chunks); print(f'Indexed {len(chunks)} chunks')"
```

### 6. Start the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

## API Endpoints

### Chat

**POST** `/api/v1/chat/query`
```json
{
  "query": "How do I configure the admin panel?",
  "use_search": true,
  "num_results": 5
}
```

**POST** `/api/v1/chat/stream` - Streaming response version

### Knowledge Management

**POST** `/api/v1/knowledge/upload` - Upload PDF document

**GET** `/api/v1/knowledge/documents` - List all documents

**DELETE** `/api/v1/knowledge/documents/{filename}` - Delete document

**POST** `/api/v1/knowledge/reindex` - Reindex all documents

### Health

**GET** `/api/v1/health` - Basic health check

**GET** `/api/v1/health/detailed` - Detailed system status

## Docker Deployment

### Build and Run with Docker Compose

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Configuration

All configuration is managed through environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | OpenAI model to use | gpt-3.5-turbo |
| `TAVILY_API_KEY` | Tavily search API key | Required |
| `CHUNK_SIZE` | Document chunk size | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 |
| `MAX_SEARCH_RESULTS` | Max web search results | 5 |

## Project Structure

```
rag-chatbot-backend/
├── app/
│   ├── api/              # API routes and models
│   ├── core/             # Core business logic (RAG, LLM, embeddings)
│   ├── services/         # External services (PDF, vector store, search)
│   ├── utils/            # Utilities (logging, etc.)
│   ├── config.py         # Configuration management
│   └── main.py           # FastAPI application
├── data/
│   ├── pdfs/             # Source PDF files
│   └── vector_db/        # ChromaDB storage
├── tests/                # Test files
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Multi-container setup
└── .env                 # Environment variables
```

## Usage Examples

### Python Client

```python
import requests

# Query the chatbot
response = requests.post(
    "http://localhost:8000/api/v1/chat/query",
    json={
        "query": "How do I reset my password?",
        "use_search": True
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {result['sources']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/v1/chat/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I configure the admin panel?"}'
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Structure

- **PDF Processing**: `app/services/pdf_processor.py`
- **Vector Store**: `app/services/vector_store.py`
- **Web Search**: `app/services/web_search.py`
- **LLM Client**: `app/core/llm_client.py`
- **RAG Pipeline**: `app/core/rag_pipeline.py`
- **API Routes**: `app/api/routes/`

## Performance Considerations

- **Embedding Model**: Uses lightweight `all-MiniLM-L6-v2` (384 dimensions) for fast inference
- **Chunking Strategy**: Smart sentence-boundary splitting with overlap for context preservation
- **Caching**: Embedding model cached as singleton
- **Batch Processing**: Efficient batch embedding generation

## Troubleshooting

### ChromaDB Issues
- Ensure `data/vector_db` directory has write permissions
- If corruption occurs, delete `data/vector_db` and reindex

### API Key Errors
- Verify `.env` file has correct API keys
- Check keys are not wrapped in quotes

### Memory Issues
- Reduce `CHUNK_SIZE` if processing large PDFs
- Limit `num_results` in queries

## Future Enhancements

- Conversation memory for multi-turn dialogues
- User authentication and API key management
- Rate limiting
- Caching layer (Redis)
- Advanced retrieval (hybrid search, re-ranking)
- Analytics dashboard
- Multi-language support

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.
