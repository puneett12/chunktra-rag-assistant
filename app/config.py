"""Single source of truth for configuration. Everything is overridable via .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b-instruct"

    # Models
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval
    candidates: int = 30
    top_k: int = 5

    # Paths
    raw_dir: str = "data/raw"
    index_dir: str = "data/processed"

    edgar_user_agent: str = "example@example.com"

    @property
    def raw_path(self) -> Path:
        return ROOT / self.raw_dir

    @property
    def index_path(self) -> Path:
        return ROOT / self.index_dir


settings = Settings()
