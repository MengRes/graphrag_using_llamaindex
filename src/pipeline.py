"""End-to-end GraphRAG pipeline orchestration."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional

from llama_index.core import Document, PropertyGraphIndex, Settings, StorageContext
from llama_index.core.graph_stores.types import EntityNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src.config import AppConfig, get_openai_api_key
from src.entity_deduplication import EntityDeduplicator
from src.extractors.graph_rag_extractor import (
    KG_TRIPLET_EXTRACT_TMPL,
    LoggingGraphRAGExtractor,
    parse_json_triplets,
)
from src.graph_utils import entity_embed_text_from_node
from src.logger import PipelineLogger, PipelineStep
from src.query_engine.graph_rag_query_engine import GraphRAGQueryEngine, QueryMode
from src.stores.graph_rag_store import LoggingGraphRAGStore


def load_graph_store_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class GraphRAGPipeline:
    """Full GraphRAG flow: load → chunk → extract → index → communities → query."""

    STEPS = [
        "1. load_documents   - Load source documents",
        "2. chunking         - Text chunking (SentenceSplitter)",
        "3. entity_extraction- LLM entity/relation extraction (GraphRAGExtractor)",
        "4. entity_deduplication - Entity dedup (normalization + embedding similarity)",
        "5. graph_index      - Build PropertyGraphIndex",
        "6. community_detection - Hierarchical Leiden community detection",
        "7. community_summary   - LLM community summaries",
        "8. persist          - Persist index and graph data",
    ]

    def __init__(self, config: AppConfig, run_id: str | None = None):
        self.config = config
        self.logger = PipelineLogger(config.logs_path, run_id=run_id)
        self._index: Optional[PropertyGraphIndex] = None
        self._graph_store: Optional[LoggingGraphRAGStore] = None
        self._llm: Optional[OpenAI] = None
        self._entity_embeddings: dict[str, list[float]] = {}

    def _setup_llm(self) -> None:
        import os

        api_key = get_openai_api_key()
        api_base = os.getenv("OPENAI_API_BASE")
        llm_kwargs: dict = {
            "model": self.config.llm.model,
            "temperature": self.config.llm.temperature,
            "api_key": api_key,
        }
        embed_kwargs: dict = {
            "model": self.config.embedding.model,
            "api_key": api_key,
        }
        if api_base:
            llm_kwargs["api_base"] = api_base
            embed_kwargs["api_base"] = api_base

        self._llm = OpenAI(**llm_kwargs)
        embed_model = OpenAIEmbedding(**embed_kwargs)
        Settings.llm = self._llm
        Settings.embed_model = embed_model

    def load_documents(self, data_dir: Path | None = None) -> list[Document]:
        self.logger.log_step_start(PipelineStep.LOAD_DOCUMENTS)
        data_path = data_dir or self.config.data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        reader = SimpleDirectoryReader(
            input_dir=str(data_path),
            required_exts=[".txt", ".md", ".pdf"],
            recursive=True,
        )
        documents = reader.load_data()

        for doc in documents:
            self.logger.log_detail(
                PipelineStep.LOAD_DOCUMENTS,
                "document_loaded",
                file_path=doc.metadata.get("file_path", "unknown"),
                text_length=len(doc.text),
            )

        self.logger.log_step_end(
            PipelineStep.LOAD_DOCUMENTS,
            document_count=len(documents),
            total_chars=sum(len(d.text) for d in documents),
        )
        return documents

    def chunk_documents(self, documents: list[Document]):
        self.logger.log_step_start(
            PipelineStep.CHUNKING,
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
        )

        # chunking documents into nodes, each node is a chunk of text. 
        # using SentenceSplitter.get_nodes_from_documents() to get nodes.
        splitter = SentenceSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents(documents)

        for i, node in enumerate(nodes):
            self.logger.log_detail(
                PipelineStep.CHUNKING,
                "chunk_created",
                chunk_index=i,
                node_id=node.node_id,
                text_length=len(node.get_content()),
                text_preview=node.get_content()[:150],
            )

        self.logger.log_step_end(
            PipelineStep.CHUNKING,
            chunk_count=len(nodes),
        )
        return nodes

    def _prune_vector_store_aliases(self, canonical_map: dict[str, str]) -> None:
        if not self._index or not canonical_map:
            return
        vector_store = self._index.vector_store
        for alias, canonical in canonical_map.items():
            if alias == canonical:
                continue
            try:
                vector_store.delete(alias)
            except Exception:
                pass

    def _refresh_index_after_dedup(self, canonical_map: dict[str, str]) -> None:
        """Sync vector store and index after in-place entity deduplication."""
        if not self._index or not self._graph_store:
            return
        self._prune_vector_store_aliases(canonical_map)
        storage_context = self._index.storage_context
        self._index = PropertyGraphIndex.from_existing(
            property_graph_store=self._graph_store,
            storage_context=storage_context,
        )

    def _build_entity_embeddings(self) -> dict[str, list[float]]:
        if not self._graph_store:
            return {}
        embed_model = Settings.embed_model
        entities = [
            n
            for n in self._graph_store.graph.nodes.values()
            if isinstance(n, EntityNode)
        ]
        if not entities:
            return {}

        names = [e.name for e in entities]
        texts = [entity_embed_text_from_node(e) for e in entities]
        vectors = embed_model.get_text_embedding_batch(texts)
        return dict(zip(names, vectors))

    def build_index(self, nodes) -> PropertyGraphIndex:
        self.logger.log_step_start(PipelineStep.ENTITY_EXTRACTION)
        self.logger.log_info(
            f"Starting entity/relation extraction for {len(nodes)} chunks, "
            f"max_paths_per_chunk={self.config.extraction.max_paths_per_chunk}"
        )

        # extract entities and relations from nodes using LoggingGraphRAGExtractor
        kg_extractor = LoggingGraphRAGExtractor(
            llm=self._llm,
            extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
            parse_fn=parse_json_triplets,
            max_paths_per_chunk=self.config.extraction.max_paths_per_chunk,
            num_workers=self.config.extraction.num_workers,
            pipeline_logger=self.logger,
        )

        self._graph_store = LoggingGraphRAGStore(
            llm=self._llm,
            max_cluster_size=self.config.community.max_cluster_size,
            pipeline_logger=self.logger,
        )

        self.logger.log_step_start(PipelineStep.GRAPH_INDEX)
        self._index = PropertyGraphIndex(
            nodes=nodes,
            kg_extractors=[kg_extractor],
            property_graph_store=self._graph_store,
            show_progress=True,
        )
        # deduplicate entities using EntityDeduplicator
        canonical_map: dict[str, str] = {}
        if self.config.deduplication.enabled:
            deduplicator = EntityDeduplicator(
                embed_model=Settings.embed_model,
                similarity_threshold=self.config.deduplication.similarity_threshold,
                pipeline_logger=self.logger,
            )
            canonical_map = deduplicator.deduplicate(self._graph_store.graph)
            self._refresh_index_after_dedup(canonical_map)

        self._entity_embeddings = self._build_entity_embeddings()

        # get triplets from graph
        triplets = self._graph_store.graph.get_triplets()
        entity_names = set()
        relations = []
        for e1, rel, e2 in triplets:
            if isinstance(e1, EntityNode):
                entity_names.add(e1.name)
            if isinstance(e2, EntityNode):
                entity_names.add(e2.name)
            relations.append(
                {
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "relation": rel.label,
                }
            )

        self.logger.log_step_end(
            PipelineStep.ENTITY_EXTRACTION,
            total_entities=len(entity_names),
            total_relations=len(relations),
            entities=sorted(entity_names)[:100],
        )
        self.logger.log_step_end(
            PipelineStep.GRAPH_INDEX,
            triplet_count=len(triplets),
            entity_count=len(entity_names),
            relation_count=len(relations),
            dedup_merges=len(canonical_map),
        )
        return self._index

    def build_communities(self) -> dict:
        if not self._graph_store:
            raise RuntimeError("Call build_index() first")
        self._graph_store.build_communities(
            summary_levels=self.config.community.summary_levels
        )
        return self._graph_store.get_community_summaries()

    def _cleanup_empty_index_files(self, index_dir: Path) -> None:
        """Remove empty placeholder files created by LlamaIndex persist."""
        empty_patterns = (
            ("graph_store.json", {"graph_dict": {}}),
            (
                "image__vector_store.json",
                {"embedding_dict": {}, "text_id_to_ref_doc_id": {}, "metadata_dict": {}},
            ),
        )
        for filename, empty_content in empty_patterns:
            path = index_dir / filename
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    if json.load(f) == empty_content:
                        path.unlink()
            except (json.JSONDecodeError, OSError):
                pass

    def _collect_alias_map(self) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        if not self._graph_store:
            return alias_map
        for node in self._graph_store.graph.nodes.values():
            if isinstance(node, EntityNode):
                for alias in node.properties.get("merged_from", []):
                    alias_map[alias] = node.name
        return alias_map

    def persist(self) -> Path:
        self.logger.log_step_start(PipelineStep.PERSIST)
        out = self.config.output_path
        out.mkdir(parents=True, exist_ok=True)

        index_dir = out / "index"
        if self._index:
            self._index.storage_context.persist(persist_dir=str(index_dir))
            self._cleanup_empty_index_files(index_dir)

        if self._graph_store:
            alias_map = self._collect_alias_map()
            store_data = {
                "community_summary_by_level": {
                    str(k): v
                    for k, v in (self._graph_store.community_summary_by_level or {}).items()
                },
                "entity_info_by_level": {
                    str(k): v
                    for k, v in (self._graph_store.entity_info_by_level or {}).items()
                },
                "cluster_assignments": self._graph_store.cluster_assignments,
            }
            if alias_map:
                store_data["entity_alias_map"] = alias_map

            with open(out / "graph_store.json", "w", encoding="utf-8") as f:
                json.dump(store_data, f, ensure_ascii=False, indent=2)

            if self._entity_embeddings:
                with open(out / "entity_embeddings.json", "w", encoding="utf-8") as f:
                    json.dump(self._entity_embeddings, f)

            with open(out / "graph_store.pkl", "wb") as f:
                pickle.dump(self._graph_store, f)

        self.logger.log_step_end(
            PipelineStep.PERSIST,
            output_dir=str(out),
        )
        return out

    # Run the full pipeline step by step.
    # 1. load_documents: load documents from data_dir
    # 2. chunk_documents: chunk documents into nodes
    # 3. build_index: build index from nodes
    # 4. build_communities: build communities from index
    # 5. persist: persist index and graph data to disk
    # 6. logger.save_summary: save summary to disk
    # 7. return index
    def run_build(self, data_dir: Path | None = None) -> PropertyGraphIndex:
        self.logger.log_step_start(PipelineStep.INIT)
        self.logger.log_info("GraphRAG Pipeline steps:")
        for step in self.STEPS:
            self.logger.log_info(f"  {step}")
        self.logger.log_step_end(PipelineStep.INIT)

        self._setup_llm()
        # load documents from data_dir
        documents = self.load_documents(data_dir)
        # chunk documents into nodes
        nodes = self.chunk_documents(documents)
        # build index from nodes
        index = self.build_index(nodes)
        # build communities from index
        self.build_communities()
        # persist index and graph data to disk
        self.persist()
        # save summary to disk
        self.logger.save_summary()
        return index

    def get_query_engine(
        self,
        mode: str | None = None,
        community_level: int | None = None,
    ) -> GraphRAGQueryEngine:
        if not self._index or not self._graph_store or not self._llm:
            raise RuntimeError("Run the build pipeline first")

        qcfg = self.config.query
        # query model: local | global | auto
        resolved_mode = QueryMode((mode or qcfg.default_mode).lower())
        level = community_level if community_level is not None else qcfg.community_level

        metadata = load_graph_store_metadata(self.config.output_path / "graph_store.json")
        entity_alias_map = metadata.get("entity_alias_map", {})
        entity_embeddings = self._entity_embeddings
        if not entity_embeddings:
            emb_path = self.config.output_path / "entity_embeddings.json"
            if emb_path.exists():
                with open(emb_path, encoding="utf-8") as f:
                    entity_embeddings = json.load(f)

        return GraphRAGQueryEngine(
            graph_store=self._graph_store,
            index=self._index,
            llm=self._llm,
            query_config=qcfg,
            mode=resolved_mode,
            community_level=level,
            pipeline_logger=self.logger,
            entity_alias_map=entity_alias_map,
            entity_embeddings=entity_embeddings,
        )

    @classmethod
    def load_from_disk(cls, config: AppConfig) -> "GraphRAGPipeline":
        """Load a previously built index from the output directory."""
        pipeline = cls(config)
        pipeline._setup_llm()

        out = config.output_path
        pkl_path = out / "graph_store.pkl"
        index_path = out / "index"
        json_path = out / "graph_store.json"

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Built index not found: {pkl_path}. Run `python main.py build` first."
            )
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index directory not found: {index_path}. Run `python main.py build` first."
            )

        try:
            with open(pkl_path, "rb") as f:
                pipeline._graph_store = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as exc:
            raise RuntimeError(
                f"Failed to load graph store from {pkl_path}. Re-run build. ({exc})"
            ) from exc

        pipeline._graph_store.attach_runtime(
            llm=pipeline._llm,
            pipeline_logger=pipeline.logger,
        )

        metadata = load_graph_store_metadata(json_path)
        if metadata:
            pipeline._graph_store.hydrate_from_metadata(metadata)

        emb_path = out / "entity_embeddings.json"
        if emb_path.exists():
            with open(emb_path, encoding="utf-8") as f:
                pipeline._entity_embeddings = json.load(f)

        storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
        pipeline._index = PropertyGraphIndex.from_existing(
            property_graph_store=pipeline._graph_store,
            storage_context=storage_context,
        )
        return pipeline
