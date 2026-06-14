"""Shared knowledge-graph helpers."""

from __future__ import annotations

import re

import networkx as nx
from llama_index.core.graph_stores.types import EntityNode


def entity_embed_text(
    name: str,
    entity_type: str = "Entity",
    description: str = "",
) -> str:
    """Canonical text for entity embedding (dedup + query)."""
    return f"{name} ({entity_type}): {description}"


def entity_embed_text_from_node(node: EntityNode) -> str:
    return entity_embed_text(
        node.name,
        node.label or "Entity",
        node.properties.get("entity_description", ""),
    )

# build entity graph from graph using NetworkX
def build_entity_graph(graph) -> nx.Graph:
    """Build a NetworkX graph from entity nodes and relations only."""
    G = nx.Graph()
    for subj, rel, obj in graph.get_triplets():
        if not isinstance(subj, EntityNode) or not isinstance(obj, EntityNode):
            continue
        G.add_node(subj.name, label=subj.name, entity_type=subj.label)
        G.add_node(obj.name, label=obj.name, entity_type=obj.label)
        G.add_edge(
            subj.name,
            obj.name,
            relation=rel.label,
            description=rel.properties.get("relationship_description", "")[:80],
        )
    return G


def clean_llm_response(raw: str) -> str:
    return re.sub(r"^assistant:\s*", "", raw).strip()
