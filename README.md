# LlamaIndex GraphRAG

A full [LlamaIndex](https://docs.llamaindex.ai/)-based GraphRAG pipeline inspired by [Microsoft GraphRAG](https://microsoft.github.io/graphrag/), from document chunking through community detection and query. **Every pipeline step writes structured JSON logs** for inspection and debugging.

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1. load_documents | `SimpleDirectoryReader` | Load txt/md/pdf from `data/` |
| 2. chunking | `SentenceSplitter` | Split text (default 512 tokens, overlap 20) |
| 3. entity_extraction | `LoggingGraphRAGExtractor` | LLM extraction of entities, relations, and descriptions |
| 4. entity_deduplication | `EntityDeduplicator` | Exact normalization + embedding-similarity merge |
| 5. graph_index | `PropertyGraphIndex` | Build property graph index |
| 6. community_detection | `hierarchical_leiden` (graspologic) | Hierarchical Leiden community detection |
| 7. community_summary | LLM | Generate a summary per community |
| 8. persist | Local storage | Persist index and graph artifacts to `output/` |
| 9. query | `GraphRAGQueryEngine` | Local / Global / Auto search (Microsoft GraphRAG style) |

## Setup

```bash
cd graphrag
pip install -r requirements.txt
```

Requires Python 3.10+ and an OpenAI-compatible API key.

## API Key

Use either option:

```bash
# Option 1: openai_key.txt (recommended, gitignored)
echo "sk-..." > openai_key.txt

# Option 2: .env
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

Default models (see `config.yaml`): `gpt-4o` for LLM, `text-embedding-3-large` for embeddings.

## Usage

### List pipeline steps

```bash
python main.py steps
```

### Build the index (full pipeline)

```bash
python main.py build
python main.py build --data-dir data/
python main.py build --run-id my-run-001   # custom log run_id
```

### Query

```bash
# Local: entity-centric search (who/what/relationships about specific entities)
python main.py query "What products does NovaMind Labs offer?" --mode local

# Global: map-reduce over community summaries (themes across the corpus)
python main.py query "What are the main themes across all documents?" --mode global

# Auto: keyword hints + LLM routing between local and global
python main.py query "How was GraphRAG used in the cybersecurity incident?" --mode auto

# Global search at a higher community hierarchy level (0 = most detailed)
python main.py query "..." --mode global --community-level 2
```

| Mode | Best for |
|------|----------|
| `local` | Entity-specific facts, relationships, named topics |
| `global` | Dataset-wide themes, trends, cross-document synthesis |
| `auto` | Convenience routing (explicit `--mode` is recommended in production) |

### Visualize the knowledge graph

```bash
python main.py visualize
python main.py visualize --level 1          # single hierarchy level (0, 1, or 2)
python main.py visualize --format png       # static PNG instead of interactive HTML
```

Outputs are written to `output/viz/` (HTML via PyVis, or PNG via matplotlib).

### Inspect logs

Logs are stored as JSON Lines in `logs/graphrag_<run_id>.jsonl` (one JSON object per line).

```bash
# Latest run
python main.py logs

# Specific run
python main.py logs --run-id 20260614_191455

# Filter by pipeline step
python main.py logs --step entity_extraction
python main.py logs --step community_detection
python main.py logs --step query

# Filter by event
python main.py logs --step entity_extraction --event chunk_extraction_done
python main.py logs --step entity_deduplication --event fuzzy_merge
python main.py logs --step community_detection --event community_detail

# Analyze with jq
cat logs/graphrag_*.jsonl | jq 'select(.step=="entity_extraction")'
cat logs/graphrag_*.jsonl | jq 'select(.event=="community_detail")'
```

## Log Events

### entity_extraction

| Event | Description |
|-------|-------------|
| `chunk_extraction_start` | Start processing a chunk |
| `chunk_extraction_done` | Extraction complete; includes entity/relation lists |
| `chunk_extraction_error` | Extraction failed |

### entity_deduplication

| Event | Description |
|-------|-------------|
| `exact_merge` | Entities merged by normalized name |
| `fuzzy_merge` | Entities merged by embedding similarity |

### community_detection

| Event | Description |
|-------|-------------|
| `graph_built` | NetworkX graph built (node/edge counts) |
| `leiden_clustering_done` | Leiden clustering assignments |
| `community_detail` | Members and relations per community |

### community_summary

| Event | Description |
|-------|-------------|
| `community_summary_generated` | LLM summary for a community (includes `level`) |

## Project Structure

```
graphrag/
├── main.py                      # CLI entry point
├── config.yaml                  # Models, chunking, community, query settings
├── .env.example                 # API key template
├── data/                        # Source documents (5 English sample docs included)
├── logs/                        # JSONL structured logs
├── output/                      # Persisted index, embeddings, graph metadata
│   └── viz/                     # Graph visualizations
└── src/
    ├── config.py                # Config loading
    ├── logger.py                # Structured logging
    ├── pipeline.py              # Pipeline orchestration
    ├── graph_utils.py           # Shared graph helpers
    ├── entity_deduplication.py  # Two-stage entity dedup
    ├── extractors/
    │   └── graph_rag_extractor.py
    ├── stores/
    │   └── graph_rag_store.py   # Leiden communities + LLM summaries
    ├── query_engine/
    │   ├── graph_rag_query_engine.py
    │   └── prompts.py
    └── visualization/
        └── graph_visualizer.py
```

## Configuration (`config.yaml`)

```yaml
llm:
  model: gpt-4o
embedding:
  model: text-embedding-3-large
chunking:
  chunk_size: 512
  chunk_overlap: 20
extraction:
  max_paths_per_chunk: 10     # Max triplets extracted per chunk
  num_workers: 4
community:
  max_cluster_size: 5         # Leiden max community size before subdivision
  summary_levels: [0, 1, 2]   # Hierarchy levels to summarize (use [0] for faster builds)
query:
  default_mode: auto          # local | global | auto
  community_level: 0          # Level used for global search (0 = most detailed)
  entity_top_k: 10
  text_chunk_top_k: 5
  map_batch_size: 5
  min_importance_score: 1
  entity_similarity_threshold: 0.3
  map_max_length: 1000
  reduce_max_length: 2000
deduplication:
  enabled: true
  similarity_threshold: 0.88  # Embedding cosine similarity threshold
paths:
  data_dir: data
  output_dir: output
  logs_dir: logs
```

## Output Artifacts

After `build`, key files under `output/`:

| File | Purpose |
|------|---------|
| `index/` | LlamaIndex persisted index (vector store, docstore, property graph) |
| `graph_store.pkl` | Serialized `LoggingGraphRAGStore` with community summaries |
| `graph_store.json` | Human-readable graph metadata and community summaries |
| `entity_embeddings.json` | Precomputed entity embeddings (used at query time) |

## References

- [LlamaIndex GraphRAG V2 Cookbook](https://docs.llamaindex.ai/en/stable/examples/cookbooks/graphrag_v2/)
- [Microsoft GraphRAG — Community Detection](https://microsoft.github.io/graphrag/concepts/community-detection/)
