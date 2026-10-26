import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# Optional, for OpenAI-compatible APIs such as OpenRouter.
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or None

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Embeddings and vector store (in memory, cleared with sessions on restart)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "webwhisper_pages"

# Retrieval and chat
TOP_K = 4
MAX_HISTORY_MESSAGES = 10
