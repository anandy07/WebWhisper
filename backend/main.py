import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from chat import reset_session, get_session, stream_answer_question
from config import HOST, PORT
from rag import index_page
from schemas import ChatRequest, IndexPageRequest, IndexPageResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WebWhisper API", version="1.0.0")


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "service": "webwhisper-api"}


@app.post("/api/v1/index", response_model=IndexPageResponse)
def index_webpage(request: IndexPageRequest):
    try:
        chunk_count, index_version = index_page(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Webpage indexing failed")
        raise HTTPException(
            status_code=500,
            detail="Webpage indexing failed. Please try again.",
        ) from exc

    reset_session(request.session_id, index_version)

    return IndexPageResponse(chunk_count=chunk_count)


@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    # Check the session before starting the HTTP stream.
    session = get_session(request.session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Page session not found. Analyze the webpage first.",
        )

    async def event_stream():
        async for event in stream_answer_question(session, request.question):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
