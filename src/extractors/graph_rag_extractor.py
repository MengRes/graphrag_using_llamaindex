"""GraphRAG entity/relation extractor with detailed logging."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, List, Optional, Union

import nest_asyncio
from llama_index.core import Settings
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import BaseNode, TransformComponent

from src.logger import PipelineLogger, PipelineStep

nest_asyncio.apply()

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Output Formatting:
- Return the result in valid JSON format with two keys: 'entities' (list of entity objects) and 'relationships' (list of relationship objects).
- Exclude any text outside the JSON structure (e.g., no explanations or comments).
- If no entities or relationships are identified, return empty lists: {{ "entities": [], "relationships": [] }}.

-An Output Example-
{{
  "entities": [
    {{
      "entity_name": "Albert Einstein",
      "entity_type": "Person",
      "entity_description": "Albert Einstein was a theoretical physicist who developed the theory of relativity."
    }}
  ],
  "relationships": [
    {{
      "source_entity": "Albert Einstein",
      "target_entity": "Theory of Relativity",
      "relation": "developed",
      "relationship_description": "Albert Einstein is the developer of the theory of relativity."
    }}
  ]
}}

-Real Data-
######################
text: {text}
######################
output:"""


def parse_json_triplets(response_str: str) -> tuple[list, list]:
    """Parse JSON-format triplets from LLM output."""
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, response_str, re.DOTALL)
    entities: list = []
    relationships: list = []
    if not match:
        return entities, relationships
    try:
        data = json.loads(match.group(0))
        entities = [
            (
                entity["entity_name"],
                entity["entity_type"],
                entity["entity_description"],
            )
            for entity in data.get("entities", [])
        ]
        relationships = [
            (
                relation["source_entity"],
                relation["target_entity"],
                relation["relation"],
                relation["relationship_description"],
            )
            for relation in data.get("relationships", [])
        ]
    except (json.JSONDecodeError, KeyError) as exc:
        import logging

        logging.getLogger(__name__).warning("Failed to parse extraction JSON: %s", exc)
    return entities, relationships


class LoggingGraphRAGExtractor(TransformComponent):
    """Extract entities and relations from text chunks with structured logging."""

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = parse_json_triplets,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
        pipeline_logger: Optional[PipelineLogger] = None,
    ) -> None:
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )
        object.__setattr__(self, "_pipeline_logger", pipeline_logger)

    @classmethod
    def class_name(cls) -> str:
        return "LoggingGraphRAGExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract(self, node: BaseNode) -> BaseNode:
        assert hasattr(node, "text")
        text = node.get_content(metadata_mode="llm")
        node_id = node.node_id

        if self._pipeline_logger:
            self._pipeline_logger.log_detail(
                PipelineStep.ENTITY_EXTRACTION,
                "chunk_extraction_start",
                node_id=node_id,
                text_preview=text[:200],
                text_length=len(text),
            )

        entities: list = []
        entities_relationship: list = []
        llm_response = ""
        error_msg = None

        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except Exception as exc:
            error_msg = str(exc)
            if self._pipeline_logger:
                self._pipeline_logger.log_detail(
                    PipelineStep.ENTITY_EXTRACTION,
                    "chunk_extraction_error",
                    node_id=node_id,
                    error=error_msg,
                )

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        for entity, entity_type, description in entities:
            entity_metadata = node.metadata.copy()
            entity_metadata["entity_description"] = description
            entity_metadata["triplet_source_id"] = node_id
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata = node.metadata.copy()
            relation_metadata["relationship_description"] = description
            relation_metadata["triplet_source_id"] = node_id
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        if self._pipeline_logger:
            self._pipeline_logger.log_detail(
                PipelineStep.ENTITY_EXTRACTION,
                "chunk_extraction_done",
                node_id=node_id,
                entity_count=len(entities),
                relation_count=len(entities_relationship),
                entities=[
                    {"name": e[0], "type": e[1], "description": e[2][:100]}
                    for e in entities
                ],
                relations=[
                    {
                        "source": r[0],
                        "target": r[1],
                        "relation": r[2],
                        "description": r[3][:100],
                    }
                    for r in entities_relationship
                ],
                error=error_msg,
            )

        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        jobs = [self._aextract(node) for node in nodes]
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )
