"""GraphRAG query engine with official-style local and global search."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.query_engine import CustomQueryEngine

from src.config import QueryConfig
from src.graph_utils import clean_llm_response, entity_embed_text_from_node
from src.logger import PipelineLogger, PipelineStep
from src.query_engine.prompts import (
    AUTO_ROUTE_PROMPT,
    DEFAULT_RESPONSE_TYPE,
    LOCAL_SEARCH_SYSTEM_PROMPT,
    MAP_SYSTEM_PROMPT,
    NO_DATA_ANSWER,
    REDUCE_SYSTEM_PROMPT,
)
from src.stores.graph_rag_store import LoggingGraphRAGStore


class QueryMode(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"
    AUTO = "auto"


@dataclass
class GraphContextIndex:
    """Pre-built indexes over graph triplets for efficient local search."""

    entities_by_name: dict[str, EntityNode] = field(default_factory=dict)
    catalog: list[tuple[str, str]] = field(default_factory=list)
    adjacency: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    relationships: list[dict] = field(default_factory=list)
    source_ids_by_entity: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    @classmethod
    def from_graph(cls, graph) -> "GraphContextIndex":
        idx = cls()
        rel_seen: set[str] = set()

        for subj, rel, obj in graph.get_triplets():
            if not isinstance(subj, EntityNode) or not isinstance(obj, EntityNode):
                continue

            for entity in (subj, obj):
                if entity.name not in idx.entities_by_name:
                    idx.entities_by_name[entity.name] = entity
                    idx.catalog.append(
                        (entity.name, entity_embed_text_from_node(entity))
                    )

            idx.adjacency[subj.name].add(obj.name)
            idx.adjacency[obj.name].add(subj.name)

            rel_key = f"{subj.name}|{rel.label}|{obj.name}"
            if rel_key not in rel_seen:
                rel_seen.add(rel_key)
                idx.relationships.append(
                    {
                        "id": len(idx.relationships) + 1,
                        "source": subj.name,
                        "target": obj.name,
                        "relation": rel.label,
                        "description": rel.properties.get("relationship_description", ""),
                    }
                )

            for entity in (subj, obj):
                sid = rel.properties.get("triplet_source_id") or entity.properties.get(
                    "triplet_source_id"
                )
                if sid:
                    idx.source_ids_by_entity[entity.name].add(sid)

        return idx


class GraphRAGQueryEngine(CustomQueryEngine):
    """Microsoft GraphRAG-style local (entity-centric) and global (map-reduce) search."""

    graph_store: LoggingGraphRAGStore
    index: PropertyGraphIndex
    llm: LLM
    query_config: QueryConfig
    mode: QueryMode = QueryMode.AUTO
    community_level: int = 0
    pipeline_logger: Optional[PipelineLogger] = None
    entity_alias_map: dict[str, str] = {}
    entity_embeddings: dict[str, list[float]] = {}

    _graph_ctx: Optional[GraphContextIndex] = None

    def custom_query(self, query_str: str) -> str:
        resolved_mode = self._resolve_mode(query_str)

        if self.pipeline_logger:
            self.pipeline_logger.log_step_start(
                PipelineStep.QUERY,
                query=query_str,
                mode=resolved_mode.value,
                community_level=self.community_level,
            )

        if resolved_mode == QueryMode.GLOBAL:
            answer = self.global_search(query_str)
        else:
            answer = self.local_search(query_str)

        if self.pipeline_logger:
            self.pipeline_logger.log_step_end(
                PipelineStep.QUERY,
                mode=resolved_mode.value,
                answer_preview=answer[:300],
            )

        return answer

    def _graph_context(self) -> GraphContextIndex:
        if self._graph_ctx is None:
            self._graph_ctx = GraphContextIndex.from_graph(self.graph_store.graph)
        return self._graph_ctx

    # resolve query mode using matching global hints in query string.
    # if query string contains global hints, return QueryMode.GLOBAL.
    # if query string does not contain global hints, return QueryMode.LOCAL.
    # This can be changed to use llm to resolve mode.
    def _resolve_mode(self, query_str: str) -> QueryMode:
        if self.mode != QueryMode.AUTO:
            return self.mode

        q = query_str.lower()
        global_hints = (
            "main theme",
            "overall",
            "across",
            "dataset",
            "top themes",
            "summarize all",
            "compare all",
        )
        if any(h in q for h in global_hints):
            return QueryMode.GLOBAL

        messages = [
            ChatMessage(
                role="user",
                content=AUTO_ROUTE_PROMPT.format(query=query_str),
            )
        ]
        response = str(self.llm.chat(messages)).strip().lower()
        token = response.split()[0].strip(".,\"'") if response.split() else ""
        if token == "global":
            return QueryMode.GLOBAL
        return QueryMode.LOCAL

    # ------------------------------------------------------------------ global
    def global_search(self, query_str: str) -> str:
        """Map-reduce over community reports (Microsoft GraphRAG global search)."""
        summaries = self.graph_store.get_community_summaries(level=self.community_level)
        if not summaries:
            return NO_DATA_ANSWER

        report_items = list(summaries.items())
        batch_size = max(1, self.query_config.map_batch_size)
        all_points: list[dict] = []

        for batch_start in range(0, len(report_items), batch_size):
            batch = report_items[batch_start : batch_start + batch_size]
            context_rows = [
                f"----Report {cid}----\n{summary}" for cid, summary in batch
            ]
            context_data = "\n\n".join(context_rows)

            map_prompt = MAP_SYSTEM_PROMPT.format(
                context_data=context_data,
                max_length=self.query_config.map_max_length,
            )
            messages = [
                ChatMessage(role="system", content=map_prompt),
                ChatMessage(role="user", content=f"Query: {query_str}"),
            ]
            raw = str(self.llm.chat(messages))
            all_points.extend(self._parse_map_points(raw))

        if not all_points:
            return NO_DATA_ANSWER

        min_score = self.query_config.min_importance_score
        filtered = [p for p in all_points if p["score"] >= min_score] or all_points
        filtered.sort(key=lambda p: p["score"], reverse=True)
        report_lines = [
            f"Analyst {i + 1} (importance {p['score']}): {p['description']}"
            for i, p in enumerate(filtered[:50])
        ]

        reduce_prompt = REDUCE_SYSTEM_PROMPT.format(
            report_data="\n\n".join(report_lines),
            response_type=self.query_config.response_type or DEFAULT_RESPONSE_TYPE,
            max_length=self.query_config.reduce_max_length,
        )
        messages = [
            ChatMessage(role="system", content=reduce_prompt),
            ChatMessage(role="user", content=f"Query: {query_str}"),
        ]
        return clean_llm_response(str(self.llm.chat(messages)))

    def _parse_map_points(self, raw: str) -> list[dict]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                parsed = []
                for pt in data.get("points", []):
                    if isinstance(pt, dict) and pt.get("description"):
                        parsed.append(
                            {
                                "description": pt["description"],
                                "score": int(pt.get("score", 50)),
                            }
                        )
                if parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        if raw.strip() and "don't know" not in raw.lower():
            return [{"description": raw.strip(), "score": 50}]
        return []

    # ------------------------------------------------------------------ local
    def local_search(self, query_str: str) -> str:
        ctx = self._graph_context()
        seed_entities = self._match_entities_by_embedding(
            query_str, self.query_config.entity_top_k, ctx
        )
        if not seed_entities:
            seed_entities = self._match_entities_by_vector_retrieval(query_str, ctx)

        expanded_entities = self._expand_entity_neighborhood(seed_entities, ctx)
        relationships = self._collect_relationships(expanded_entities, ctx)
        text_chunks = self._collect_text_chunks(expanded_entities, query_str, ctx)
        community_reports = self._collect_community_reports(seed_entities)

        if not expanded_entities and not text_chunks and not community_reports:
            return NO_DATA_ANSWER

        context_data = self._build_local_context_tables(
            expanded_entities,
            relationships,
            text_chunks,
            community_reports,
        )

        system_prompt = LOCAL_SEARCH_SYSTEM_PROMPT.format(
            context_data=context_data,
            response_type=self.query_config.response_type or DEFAULT_RESPONSE_TYPE,
        )
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=query_str),
        ]
        return clean_llm_response(str(self.llm.chat(messages)))

    def _match_entities_by_embedding(
        self,
        query_str: str,
        top_k: int,
        ctx: GraphContextIndex,
    ) -> list[str]:
        if not ctx.catalog:
            return []

        names = [name for name, _ in ctx.catalog]
        threshold = self.query_config.entity_similarity_threshold

        if self.entity_embeddings:
            query_emb = np.array(Settings.embed_model.get_query_embedding(query_str))
            query_norm = np.linalg.norm(query_emb)
            if query_norm == 0:
                return []
            vectors = []
            valid_names = []
            for name in names:
                emb = self.entity_embeddings.get(name)
                if emb:
                    vectors.append(emb)
                    valid_names.append(name)
            if not vectors:
                return []
            text_embs = np.array(vectors)
            scores = text_embs @ query_emb / (
                np.linalg.norm(text_embs, axis=1) * query_norm + 1e-9
            )
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [valid_names[i] for i in top_indices if scores[i] > threshold]

        embed_model = Settings.embed_model
        texts = [text for _, text in ctx.catalog]
        query_emb = np.array(embed_model.get_query_embedding(query_str))
        text_embs = np.array(embed_model.get_text_embedding_batch(texts))
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return []
        scores = text_embs @ query_emb / (
            np.linalg.norm(text_embs, axis=1) * query_norm + 1e-9
        )
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [names[i] for i in top_indices if scores[i] > threshold]

    def _match_entities_by_vector_retrieval(
        self, query_str: str, ctx: GraphContextIndex
    ) -> list[str]:
        retriever = self.index.as_retriever(
            similarity_top_k=self.query_config.text_chunk_top_k
        )
        nodes = retriever.retrieve(query_str)
        catalog_names = {name.lower(): name for name, _ in ctx.catalog}

        found: list[str] = []
        for node in nodes:
            text_lower = node.text.lower()
            for key, canonical in sorted(catalog_names.items(), key=lambda x: -len(x[0])):
                pattern = r"\b" + re.escape(key) + r"\b"
                if re.search(pattern, text_lower) and canonical not in found:
                    found.append(canonical)
        return found[: self.query_config.entity_top_k]

    def _resolve_entity_name(self, name: str) -> str:
        return self.entity_alias_map.get(name, name)

    def _expand_entity_neighborhood(
        self, seed_entities: list[str], ctx: GraphContextIndex
    ) -> list[dict]:
        seed_set = {self._resolve_entity_name(e) for e in seed_entities}
        expanded_names = set(seed_set)
        for name in seed_set:
            expanded_names.update(ctx.adjacency.get(name, ()))

        expanded: dict[str, dict] = {}
        for name in expanded_names:
            entity = ctx.entities_by_name.get(name)
            if entity:
                expanded[name] = {
                    "id": name,
                    "name": name,
                    "type": entity.label or "Entity",
                    "description": entity.properties.get("entity_description", ""),
                }
        return list(expanded.values())

    def _collect_relationships(
        self, entities: list[dict], ctx: GraphContextIndex
    ) -> list[dict]:
        entity_names = {e["name"] for e in entities}
        return [
            r
            for r in ctx.relationships
            if r["source"] in entity_names or r["target"] in entity_names
        ]

    def _collect_text_chunks(
        self,
        entities: list[dict],
        query_str: str,
        ctx: GraphContextIndex,
    ) -> list[dict]:
        entity_names = {e["name"] for e in entities}
        source_ids: set[str] = set()
        for name in entity_names:
            source_ids.update(ctx.source_ids_by_entity.get(name, ()))

        chunks: list[dict] = []
        for sid in source_ids:
            try:
                node = self.index.docstore.get_node(sid)
                if node:
                    chunks.append({"id": sid, "text": node.get_content()})
            except (KeyError, ValueError):
                continue

        if len(chunks) < self.query_config.text_chunk_top_k:
            retriever = self.index.as_retriever(
                similarity_top_k=self.query_config.text_chunk_top_k
            )
            for hit in retriever.retrieve(query_str):
                nid = hit.node.node_id
                if not any(c["id"] == nid for c in chunks):
                    chunks.append({"id": nid, "text": hit.text})

        return chunks[: self.query_config.text_chunk_top_k]

    def _collect_community_reports(self, entities: list[str]) -> list[dict]:
        entity_info = self.graph_store.get_entity_info(level=self.community_level)
        summaries = self.graph_store.get_community_summaries(level=self.community_level)
        community_ids: set = set()

        for entity in entities:
            resolved = self._resolve_entity_name(entity)
            for cid in entity_info.get(resolved, []):
                community_ids.add(cid)

        return [
            {"id": cid, "summary": summaries[cid]}
            for cid in sorted(community_ids)
            if cid in summaries
        ]

    def _build_local_context_tables(
        self,
        entities: list[dict],
        relationships: list[dict],
        text_chunks: list[dict],
        community_reports: list[dict],
    ) -> str:
        sections = []

        if entities:
            rows = [
                f"id: {e['id']}, name: {e['name']}, type: {e['type']}, "
                f"description: {e['description']}"
                for e in entities
            ]
            sections.append("-----Entities-----\n" + "\n".join(rows))

        if relationships:
            rows = [
                f"id: {r['id']}, source: {r['source']}, target: {r['target']}, "
                f"relation: {r['relation']}, description: {r['description']}"
                for r in relationships
            ]
            sections.append("-----Relationships-----\n" + "\n".join(rows))

        if text_chunks:
            rows = [f"id: {c['id']}, text: {c['text'][:800]}" for c in text_chunks]
            sections.append("-----Sources-----\n" + "\n".join(rows))

        if community_reports:
            rows = [f"id: {r['id']}, summary: {r['summary']}" for r in community_reports]
            sections.append("-----Reports-----\n" + "\n".join(rows))

        return "\n\n".join(sections)
