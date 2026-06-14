"""GraphRAG graph store with community detection and summary logging."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from graspologic.partition import hierarchical_leiden
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.llms import ChatMessage, LLM

from src.graph_utils import build_entity_graph, clean_llm_response
from src.logger import PipelineLogger, PipelineStep


class LoggingGraphRAGStore(SimplePropertyGraphStore):
    """SimplePropertyGraphStore + hierarchical Leiden communities + LLM summaries."""

    max_cluster_size: int = 5

    def __init__(
        self,
        llm: LLM,
        max_cluster_size: int = 5,
        pipeline_logger: Optional[PipelineLogger] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm = llm
        self.max_cluster_size = max_cluster_size
        self._pipeline_logger = pipeline_logger
        self.community_summary: dict = {}
        self.community_summary_by_level: dict = {}
        self.entity_info: dict | None = None
        self.entity_info_by_level: dict = {}
        self.cluster_assignments: list = []
        self._communities_built = False

    def attach_runtime(
        self,
        llm: LLM | None = None,
        pipeline_logger: PipelineLogger | None = None,
    ) -> None:
        if llm is not None:
            self._llm = llm
        if pipeline_logger is not None:
            self._pipeline_logger = pipeline_logger

    def generate_community_summary(self, text: str) -> str:
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = self._llm.chat(messages)
        return clean_llm_response(str(response))


    # build communities from graph using hierarchical Leiden community detection
    def build_communities(
        self, summary_levels: list[int] | None = None
    ) -> None:
        if self._pipeline_logger:
            self._pipeline_logger.log_step_start(PipelineStep.COMMUNITY_DETECTION)

        nx_graph = build_entity_graph(self.graph)
        node_count = nx_graph.number_of_nodes()
        edge_count = nx_graph.number_of_edges()

        if self._pipeline_logger:
            self._pipeline_logger.log_detail(
                PipelineStep.COMMUNITY_DETECTION,
                "graph_built",
                node_count=node_count,
                edge_count=edge_count,
                nodes=list(nx_graph.nodes())[:50],
            )

        if node_count == 0:
            self._communities_built = True
            if self._pipeline_logger:
                self._pipeline_logger.log_step_end(
                    PipelineStep.COMMUNITY_DETECTION,
                    community_count=0,
                    message="Graph is empty; skipping community detection",
                )
            return

        clusters = hierarchical_leiden(nx_graph, max_cluster_size=self.max_cluster_size)
        self.cluster_assignments = [
            {"node": item.node, "cluster": item.cluster, "level": item.level}
            for item in clusters
        ]
        levels = sorted({item.level for item in clusters})
        levels_to_summarize = (
            [lv for lv in (summary_levels or levels) if lv in levels]
            or levels
        )

        if self._pipeline_logger:
            self._pipeline_logger.log_detail(
                PipelineStep.COMMUNITY_DETECTION,
                "leiden_clustering_done",
                algorithm="hierarchical_leiden",
                max_cluster_size=self.max_cluster_size,
                community_count=len({item.cluster for item in clusters}),
                levels=levels,
                summary_levels=levels_to_summarize,
                assignment_count=len(clusters),
                assignments=[
                    {"node": item.node, "cluster": item.cluster, "level": item.level}
                    for item in clusters[:40]
                ],
            )

        self.entity_info_by_level = {}
        self.community_summary_by_level = {}

        if self._pipeline_logger:
            self._pipeline_logger.log_step_start(PipelineStep.COMMUNITY_SUMMARY)

        for level in levels_to_summarize:
            entity_info_lvl, community_info_lvl = self._collect_community_info(
                nx_graph, clusters, level=level
            )
            self.entity_info_by_level[level] = entity_info_lvl

            if level == 0:
                self.entity_info = entity_info_lvl
                if self._pipeline_logger:
                    for cid, details in community_info_lvl.items():
                        self._pipeline_logger.log_detail(
                            PipelineStep.COMMUNITY_DETECTION,
                            "community_detail",
                            community_id=cid,
                            member_count=len(
                                [n for n, cs in entity_info_lvl.items() if cid in cs]
                            ),
                            relation_count=len(details),
                            relations=details[:20],
                        )

            summaries: dict = {}
            for community_id, details in community_info_lvl.items():
                details_text = "\n".join(details) + "."
                summaries[community_id] = self.generate_community_summary(details_text)
            self.community_summary_by_level[level] = summaries

            if self._pipeline_logger:
                for cid, summary in summaries.items():
                    self._pipeline_logger.log_detail(
                        PipelineStep.COMMUNITY_SUMMARY,
                        "community_summary_generated",
                        community_id=cid,
                        level=level,
                        summary=summary,
                    )

        self.community_summary = self.community_summary_by_level.get(0, {})
        self._communities_built = True

        if self._pipeline_logger:
            self._pipeline_logger.log_step_end(
                PipelineStep.COMMUNITY_DETECTION,
                community_count=len(
                    self.community_summary_by_level.get(0, {})
                ),
                total_entities=len(self.entity_info or {}),
                levels=levels,
            )
            self._pipeline_logger.log_step_end(
                PipelineStep.COMMUNITY_SUMMARY,
                summary_count=sum(
                    len(v) for v in self.community_summary_by_level.values()
                ),
                levels=levels_to_summarize,
            )

    def _collect_community_info(
        self, nx_graph, clusters, level: int = 0
    ) -> tuple[dict, dict]:
        entity_info: dict = defaultdict(set)
        community_info: dict = defaultdict(list)
        level_nodes = {item.node for item in clusters if item.level == level}
        seen_edges: set[tuple] = set()

        for item in clusters:
            if item.level != level:
                continue
            node = item.node
            cluster_id = item.cluster
            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                if neighbor not in level_nodes:
                    continue
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if not edge_data:
                    continue
                relation = edge_data.get("relationship", "")
                edge_key = tuple(sorted((node, neighbor))) + (relation,)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                detail = (
                    f"{node} -> {neighbor} -> "
                    f"{relation} -> {edge_data.get('description', '')}"
                )
                community_info[cluster_id].append(detail)

        entity_info = {k: list(v) for k, v in entity_info.items()}
        return dict(entity_info), dict(community_info)

    # get community summaries by level
    def get_community_summaries(self, level: int = 0) -> dict:
        if not self._communities_built:
            raise RuntimeError(
                "Communities are not built. Run the build pipeline first."
            )
        if self.community_summary_by_level:
            return self.community_summary_by_level.get(
                level, self.community_summary_by_level.get(0, {})
            )
        return self.community_summary

    def get_entity_info(self, level: int = 0) -> dict:
        if not self._communities_built:
            raise RuntimeError(
                "Communities are not built. Run the build pipeline first."
            )
        if self.entity_info_by_level:
            return self.entity_info_by_level.get(
                level, self.entity_info_by_level.get(0, {})
            )
        return self.entity_info or {}

    def hydrate_from_metadata(self, metadata: dict) -> None:
        """Restore community metadata loaded from graph_store.json."""
        by_level = metadata.get("community_summary_by_level")
        if by_level:
            self.community_summary_by_level = {
                int(k): v for k, v in by_level.items()
            }
            self.community_summary = self.community_summary_by_level.get(
                0, metadata.get("community_summary", {})
            )
        elif metadata.get("community_summary"):
            self.community_summary = metadata["community_summary"]
            self.community_summary_by_level = {0: self.community_summary}

        entity_by_level = metadata.get("entity_info_by_level")
        if entity_by_level:
            self.entity_info_by_level = {int(k): v for k, v in entity_by_level.items()}
            self.entity_info = self.entity_info_by_level.get(
                0, metadata.get("entity_info")
            )
        elif metadata.get("entity_info"):
            self.entity_info = metadata["entity_info"]
            self.entity_info_by_level = {0: self.entity_info}

        if metadata.get("cluster_assignments"):
            self.cluster_assignments = metadata["cluster_assignments"]

        if self.community_summary_by_level or self.community_summary:
            self._communities_built = True
