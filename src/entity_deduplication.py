"""Entity deduplication via normalization + embedding similarity merging."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    LabelledPropertyGraph,
    Relation,
)

from src.graph_utils import entity_embed_text_from_node
from src.logger import PipelineLogger, PipelineStep


def normalize_entity_name(name: str) -> str:
    """Normalize entity names for exact deduplication."""
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name.lower()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _is_word_boundary_prefix(shorter: str, longer: str) -> bool:
    """Merge only when the shorter name is a word-boundary prefix of the longer name."""
    if len(shorter) >= len(longer):
        return False
    if not longer.lower().startswith(shorter.lower()):
        return False
    suffix = longer[len(shorter) :]
    if not suffix:
        return True
    return suffix[0] in (" ", "-", "_", "'")


def _should_merge(
    a: str, b: str, similarity: float, threshold: float
) -> bool:
    a, b = a.strip(), b.strip()
    if normalize_entity_name(a) == normalize_entity_name(b):
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if _is_word_boundary_prefix(short, long):
        return True
    if similarity >= threshold:
        return True
    return False


def _pick_canonical(names: list[str]) -> str:
    """Pick the longest, most specific name as the canonical form."""
    return max(names, key=lambda n: (len(n), n))


def _merge_entity_nodes(nodes: list[EntityNode], canonical: str) -> EntityNode:
    labels = [n.label for n in nodes if n.label and n.label != "entity"]
    label = Counter(labels).most_common(1)[0][0] if labels else nodes[0].label

    descriptions = [
        n.properties.get("entity_description", "")
        for n in nodes
        if n.properties.get("entity_description")
    ]
    merged_desc = descriptions[0] if descriptions else ""
    if len(descriptions) > 1:
        merged_desc = max(descriptions, key=len)

    embedding = next((n.embedding for n in nodes if n.embedding), None)
    properties = dict(nodes[0].properties)
    properties["entity_description"] = merged_desc
    properties["merged_from"] = sorted({n.name for n in nodes if n.name != canonical})

    return EntityNode(
        name=canonical,
        label=label,
        properties=properties,
        embedding=embedding,
    )


class EntityDeduplicator:
    """Deduplicate EntityNode instances in a PropertyGraph and rebuild relations."""

    def __init__(
        self,
        embed_model: BaseEmbedding,
        similarity_threshold: float = 0.88,
        pipeline_logger: Optional[PipelineLogger] = None,
    ):
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.logger = pipeline_logger

    def deduplicate(self, graph: LabelledPropertyGraph) -> dict[str, str]:
        """
        Deduplicate entities in place.

        Returns:
            Mapping of alias -> canonical name.
        """
        entities = [
            n for n in graph.nodes.values() if isinstance(n, EntityNode)
        ]
        if not entities:
            return {}

        if self.logger:
            self.logger.log_step_start(
                PipelineStep.ENTITY_DEDUPLICATION,
                raw_entity_count=len(entities),
                similarity_threshold=self.similarity_threshold,
            )

        # Tier 1: exact normalization merge
        norm_groups: dict[str, list[str]] = defaultdict(list)
        for e in entities:
            norm_groups[normalize_entity_name(e.name)].append(e.name)

        canonical_map: dict[str, str] = {}
        exact_merges: list[dict] = []
        for names in norm_groups.values():
            unique_names = list(dict.fromkeys(names))
            canonical = _pick_canonical(unique_names)
            for name in unique_names:
                canonical_map[name] = canonical
            if len(unique_names) > 1:
                exact_merges.append(
                    {"canonical": canonical, "merged": unique_names}
                )

        # Tier 2: embedding-based fuzzy merge (name + type + description)
        entity_by_canonical: dict[str, EntityNode] = {}
        for e in entities:
            canon = canonical_map.get(e.name, e.name)
            if canon not in entity_by_canonical:
                entity_by_canonical[canon] = e

        unique_canonicals = list(entity_by_canonical.keys())
        embed_texts = [
            entity_embed_text_from_node(entity_by_canonical[c])
            for c in unique_canonicals
        ]
        embeddings = self.embed_model.get_text_embedding_batch(embed_texts)

        parent = {c: c for c in unique_canonicals}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            pa, pb = find(a), find(b)
            if pa == pb:
                return
            longer = _pick_canonical([pa, pb])
            shorter = pb if longer == pa else pa
            parent[shorter] = longer

        fuzzy_merges: list[dict] = []
        for i in range(len(unique_canonicals)):
            for j in range(i + 1, len(unique_canonicals)):
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                a, b = unique_canonicals[i], unique_canonicals[j]
                if _should_merge(a, b, sim, self.similarity_threshold):
                    if find(a) != find(b):
                        fuzzy_merges.append(
                            {
                                "canonical": _pick_canonical([a, b]),
                                "merged": [a, b],
                                "similarity": round(sim, 4),
                            }
                        )
                    union(a, b)

        for name in list(canonical_map.keys()):
            canonical_map[name] = find(canonical_map[name])

        # Rebuild graph (preserve ChunkNodes)
        entity_by_name = {e.name: e for e in entities}
        new_graph = LabelledPropertyGraph()

        for node in graph.nodes.values():
            if isinstance(node, ChunkNode):
                new_graph.add_node(node)

        merged_groups: dict[str, list[EntityNode]] = defaultdict(list)
        for e in entities:
            canon = canonical_map.get(e.name, e.name)
            merged_groups[canon].append(entity_by_name[e.name])

        for canon, group in merged_groups.items():
            new_graph.add_node(_merge_entity_nodes(group, canon))

        seen_relations: set[tuple[str, str, str]] = set()
        for rel in graph.relations.values():
            src = canonical_map.get(rel.source_id, rel.source_id)
            tgt = canonical_map.get(rel.target_id, rel.target_id)
            if src == tgt:
                continue
            key = (src, rel.label, tgt)
            if key in seen_relations:
                continue
            seen_relations.add(key)
            new_rel = Relation(
                label=rel.label,
                source_id=src,
                target_id=tgt,
                properties=dict(rel.properties),
            )
            new_graph.add_relation(new_rel)

        # Replace graph in place
        graph.nodes = new_graph.nodes
        graph.relations = new_graph.relations
        graph.triplets = new_graph.triplets

        deduped_count = len(merged_groups)
        if self.logger:
            for m in exact_merges:
                self.logger.log_detail(
                    PipelineStep.ENTITY_DEDUPLICATION,
                    "exact_merge",
                    **m,
                )
            for m in fuzzy_merges:
                self.logger.log_detail(
                    PipelineStep.ENTITY_DEDUPLICATION,
                    "fuzzy_merge",
                    **m,
                )
            self.logger.log_step_end(
                PipelineStep.ENTITY_DEDUPLICATION,
                raw_entity_count=len(entities),
                deduped_entity_count=deduped_count,
                merged_count=len(entities) - deduped_count,
                exact_merge_groups=len(exact_merges),
                fuzzy_merge_pairs=len(fuzzy_merges),
            )

        return canonical_map
