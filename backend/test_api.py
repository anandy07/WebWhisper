import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import chat
import rag
from main import app

client = TestClient(app)
SESSION = "test-session-0001"
PAGE = {
    "session_id": SESSION,
    "page_url": "https://example.com/coffee",
    "page_title": "Coffee Guide",
    "sources": [
        {
            "heading": "Brewing",
            "text": "Use water at 93 degrees Celsius for pour-over coffee.",
        },
        {
            "heading": "Storage",
            "text": "Keep beans in an airtight container away from light.",
        },
    ],
}


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(
        chat, "get_chat_model", lambda: FakeListChatModel(responses=["Use 93 degrees."])
    )


def ask(question, session_id=SESSION):
    response = client.post(
        "/api/v1/chat", json={"session_id": session_id, "question": question}
    )
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    return response, events


def test_health():
    assert client.get("/api/v1/health").json()["status"] == "ok"


def test_index_and_chat(fake_llm):
    assert client.post("/api/v1/index", json=PAGE).json()["chunk_count"] == 2

    _, events = ask("What water temperature should I use?")
    answer = "".join(e["content"] for e in events if e["type"] == "answer_delta")
    citations = next(e["citations"] for e in events if e["type"] == "citations")

    assert answer == "Use 93 degrees."
    assert citations[0]["heading"] == "Brewing"
    assert events[-1]["type"] == "done"
    assert len(chat.get_session(SESSION).history) == 2


def test_reindex_resets_history(fake_llm):
    client.post("/api/v1/index", json=PAGE)
    ask("Hi")
    client.post("/api/v1/index", json=PAGE)
    assert chat.get_session(SESSION).history == []


def test_chat_unknown_session():
    response, _ = ask("Hello?", session_id="missing-session-01")
    assert response.status_code == 404


def test_index_rejects_blank_text():
    page = {**PAGE, "sources": [{"text": "   "}]}
    assert client.post("/api/v1/index", json=page).status_code == 422


def test_missing_llm_config_is_reported(monkeypatch):
    client.post("/api/v1/index", json=PAGE)
    monkeypatch.setattr(
        chat,
        "get_chat_model",
        lambda: (_ for _ in ()).throw(chat.LLMConfigError("LLM is not configured.")),
    )
    _, events = ask("Anything?")
    assert events == [
        {
            "type": "error",
            "code": "llm_unavailable",
            "message": "LLM is not configured.",
        }
    ]


def test_unexpected_runtime_error_is_not_config_error(monkeypatch):
    client.post("/api/v1/index", json=PAGE)
    monkeypatch.setattr(
        chat, "get_chat_model", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    _, events = ask("Anything?")
    assert events[-1]["code"] == "chat_failed"


def test_failed_old_index_cleanup_does_not_mix_pages(fake_llm, monkeypatch):
    client.post("/api/v1/index", json=PAGE)

    # Simulate the old-version cleanup failing on re-index.
    store = rag.get_vector_store()
    real_delete = store.delete

    def failing_delete(*args, **kwargs):
        if "$ne" in str(kwargs.get("where")):
            raise RuntimeError("delete failed")
        return real_delete(*args, **kwargs)

    monkeypatch.setattr(store, "delete", failing_delete)
    new_page = {
        **PAGE,
        "page_title": "Tea Guide",
        "sources": [{"heading": "Steeping", "text": "Steep green tea for two minutes."}],
    }
    assert client.post("/api/v1/index", json=new_page).status_code == 200

    _, events = ask("What water temperature should I use?")
    citations = next(e["citations"] for e in events if e["type"] == "citations")
    assert {c["page_title"] for c in citations} == {"Tea Guide"}
