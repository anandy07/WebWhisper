from pydantic import BaseModel, Field, HttpUrl


class PageSource(BaseModel):
    heading: str | None = None
    text: str = Field(min_length=1)


class IndexPageRequest(BaseModel):
    session_id: str = Field(min_length=16, max_length=100)
    page_url: HttpUrl
    page_title: str
    sources: list[PageSource] = Field(min_length=1)


class IndexPageResponse(BaseModel):
    chunk_count: int


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=16, max_length=100)
    question: str = Field(min_length=1, max_length=5000)
