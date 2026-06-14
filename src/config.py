"""GraphRAG configuration loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-large"


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 20


@dataclass
class ExtractionConfig:
    max_paths_per_chunk: int = 10
    num_workers: int = 4


@dataclass
class DeduplicationConfig:
    enabled: bool = True
    similarity_threshold: float = 0.88


@dataclass
class CommunityConfig:
    max_cluster_size: int = 5
    summary_levels: list[int] | None = None


@dataclass
class QueryConfig:
    default_mode: str = "auto"
    community_level: int = 0
    entity_top_k: int = 10
    text_chunk_top_k: int = 5
    map_max_length: int = 1000
    reduce_max_length: int = 2000
    map_batch_size: int = 5
    min_importance_score: int = 1
    entity_similarity_threshold: float = 0.3
    response_type: str = "Multiple Paragraphs"


@dataclass
class PathsConfig:
    data_dir: str = "data"
    output_dir: str = "output"
    logs_dir: str = "logs"


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    community: CommunityConfig = field(default_factory=CommunityConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    project_root: Path = field(default_factory=Path.cwd)

    @property
    def data_path(self) -> Path:
        return self.project_root / self.paths.data_dir

    @property
    def output_path(self) -> Path:
        return self.project_root / self.paths.output_dir

    @property
    def logs_path(self) -> Path:
        return self.project_root / self.paths.logs_dir


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load config from config.yaml and read API key from .env."""
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    cfg_file = Path(config_path) if config_path else root / "config.yaml"
    raw: dict = {}
    if cfg_file.exists():
        with open(cfg_file, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    def _section(name: str, cls: type):
        return cls(**(raw.get(name) or {}))

    return AppConfig(
        llm=_section("llm", LLMConfig),
        embedding=_section("embedding", EmbeddingConfig),
        chunking=_section("chunking", ChunkingConfig),
        extraction=_section("extraction", ExtractionConfig),
        deduplication=_section("deduplication", DeduplicationConfig),
        community=_section("community", CommunityConfig),
        query=_section("query", QueryConfig),
        paths=_section("paths", PathsConfig),
        project_root=root,
    )


def get_openai_api_key(project_root: Path | None = None) -> str:
    """Read API key. Priority: env var > .env > openai_key.txt."""
    root = project_root or Path(__file__).resolve().parent.parent

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key and not key.startswith("sk-your"):
        return key

    key_file = root / "openai_key.txt"
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key

    raise ValueError(
        "OPENAI_API_KEY is not set. Configure it in openai_key.txt, .env, or an environment variable."
    )
