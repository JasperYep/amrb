from pydantic import BaseModel, ConfigDict, Field, computed_field


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


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    file_name: str
    pages: int | None = None
    chunks: int = 0


class PaperFailure(BaseModel):
    source: str
    message: str
    copied: bool = False


class PaperAddResult(BaseModel):
    added: list[PaperRecord] = Field(default_factory=list)
    updated: list[PaperRecord] = Field(default_factory=list)
    skipped: list[PaperRecord] = Field(default_factory=list)
    failed: list[PaperFailure] = Field(default_factory=list)


class PaperSyncResult(BaseModel):
    added: int
    updated: int
    removed: int
    unchanged: int
    papers: int
    chunks: int
    error: str | None = None


class PaperRemoveResult(BaseModel):
    removed: PaperRecord


class ComponentHealth(BaseModel):
    status: str
    message: str | None = None


class SystemStatus(BaseModel):
    paper_rag: str
    hindsight: str
    qdrant: str
    papers: int
    chunks: int


class DoctorReport(BaseModel):
    environment: ComponentHealth
    embedding: ComponentHealth
    qdrant: ComponentHealth
    paper_index: ComponentHealth
    hindsight: ComponentHealth
    memory_bank: ComponentHealth
    papers: int
    chunks: int
