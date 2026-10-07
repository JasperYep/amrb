from pydantic import BaseModel, ConfigDict, computed_field


class PaperHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    paper_id: str
    title: str
    file_name: str
    page: int
    score: float
    text: str

    @computed_field
    @property
    def citation(self) -> str:
        label = self.title.strip() or self.file_name
        return f"{label}, p. {self.page}"


class PaperBrief(BaseModel):
    paper_id: str
    title: str
    file_name: str


class PaperPage(BaseModel):
    paper_id: str
    title: str
    file_name: str
    page: int
    citation: str
    text: str


class IngestResult(BaseModel):
    discovered: int
    indexed: int
    skipped: int
    removed: int
    chunks: int


class PaperRAGStatus(BaseModel):
    papers: int
    chunks: int
    vector_backend: str
    collection: str
    last_ingest: str | None
