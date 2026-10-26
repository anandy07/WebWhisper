# WebWhisper

Chrome extension that lets you chat with any webpage. The page text is chunked, embedded locally (MiniLM), stored in an in-memory ChromaDB, and answered by the LLM of your choice with source headings.

```
backend/    FastAPI RAG API (main.py, rag.py, chat.py, schemas.py, config.py)
extension/  Chrome MV3 side panel (manifest.json, sidepanel.html/js, background.js)
``` 


## Setup

1. Backend:
   ```bash
   cd backend
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   .venv/bin/pip install langchain-anthropic   # or langchain-openai, langchain-google-genai, ...
   cp .env.example .env                        # set LLM_PROVIDER, LLM_MODEL, LLM_API_KEY
   .venv/bin/python main.py                    # serves http://127.0.0.1:8000
   ```
   Example `.env`: `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-sonnet-5`.
   OpenRouter: `LLM_PROVIDER=openai`, `LLM_BASE_URL=https://openrouter.ai/api/v1`, `LLM_MODEL=<model id>` (needs `langchain-openai`).

2. Extension: open `chrome://extensions`, enable Developer mode, click **Load unpacked**, select `extension/`.

3. Open any page, click the WebWhisper icon, press **Analyze page**, ask questions.

## API

| Method | Path | Body |
|---|---|---|
| GET | `/api/v1/health` | |
| POST | `/api/v1/index` | `{session_id, page_url, page_title, sources: [{heading, text}]}` |
| POST | `/api/v1/chat` | `{session_id, question}`, streams SSE events `answer_delta`, `citations`, `done`, `error` |

## Tests

```bash
cd backend
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

## Notes

- Sessions and vectors live in memory; restarting the backend clears them (re-analyze the page).
- Chrome blocks extensions from reading `chrome://` pages and the Web Store.
