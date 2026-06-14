"""Structured JSON logging for the GraphRAG pipeline.

Each pipeline step writes JSON Lines records for grep / jq querying.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class PipelineStep(str, Enum):
    """Pipeline step identifiers."""

    INIT = "init"
    LOAD_DOCUMENTS = "load_documents"
    CHUNKING = "chunking"
    ENTITY_EXTRACTION = "entity_extraction"
    ENTITY_DEDUPLICATION = "entity_deduplication"
    GRAPH_INDEX = "graph_index"
    COMMUNITY_DETECTION = "community_detection"
    COMMUNITY_SUMMARY = "community_summary"
    PERSIST = "persist"
    QUERY = "query"


class PipelineLogger:
    """Dual-channel logger: console + JSONL file."""

    def __init__(self, logs_dir: Path, run_id: str | None = None):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or ts
        self.jsonl_path = self.logs_dir / f"graphrag_{self.run_id}.jsonl"
        self.summary_path = self.logs_dir / f"graphrag_{self.run_id}_summary.json"

        self._console = logging.getLogger("graphrag")
        if not self._console.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            self._console.addHandler(handler)
            self._console.setLevel(logging.INFO)

        self._step_stats: dict[str, Any] = {}

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        record.setdefault("run_id", self.run_id)
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_step_start(self, step: PipelineStep, **extra: Any) -> None:
        msg = f"[{step.value}] started"
        self._console.info(msg)
        self._write_jsonl(
            {"event": "step_start", "step": step.value, **extra}
        )

    def log_step_end(self, step: PipelineStep, **extra: Any) -> None:
        msg = f"[{step.value}] completed"
        self._console.info(msg)
        self._step_stats[step.value] = extra
        self._write_jsonl({"event": "step_end", "step": step.value, **extra})

    def log_step_error(self, step: PipelineStep, error: str, **extra: Any) -> None:
        self._console.error(f"[{step.value}] failed: {error}")
        self._write_jsonl(
            {"event": "step_error", "step": step.value, "error": error, **extra}
        )

    def log_detail(self, step: PipelineStep, event: str, **extra: Any) -> None:
        """Log a fine-grained event within a step (extraction, community detection, etc.)."""
        self._write_jsonl(
            {"event": event, "step": step.value, **extra}
        )

    def log_info(self, message: str, **extra: Any) -> None:
        self._console.info(message)
        if extra:
            self._write_jsonl({"event": "info", "message": message, **extra})

    def save_summary(self) -> Path:
        summary = {
            "run_id": self.run_id,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "steps": self._step_stats,
            "log_file": str(self.jsonl_path),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return self.summary_path
