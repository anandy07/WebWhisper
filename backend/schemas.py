from pydantic import BaseModel, Field, HttpUrl, model_validator

from config import MAX_HEADING_CHARS, MAX_PAGE_CHARS, MAX_SOURCES, MAX_TITLE_CHARS


class PageSource(BaseModel):
    heading: str | None = Field(default=None, max_length=MAX_HEADING_CHARS)
    text: str = Field(min_length=1, max_length=MAX_PAGE_CHARS)


class IndexPageRequest(BaseModel):
    session_id: str = Field(min_length=16, max_length=100)
    page_url: HttpUrl
    page_title: str = Field(max_length=MAX_TITLE_CHARS)
    sources: list[PageSource] = Field(min_length=1, max_length=MAX_SOURCES)

    @model_validator(mode="after")
    def check_total_size(self):
        total = sum(len(source.text) for source in self.sources)
        if total > MAX_PAGE_CHARS:
            raise ValueError(
                f"Page is too large to analyze ({total:,} characters, "
                f"limit {MAX_PAGE_CHARS:,})."
            )
        return self


class IndexPageResponse(BaseModel):
    chunk_count: int


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=16, max_length=100)
    question: str = Field(min_length=1, max_length=5000)
