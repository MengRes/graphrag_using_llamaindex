#!/usr/bin/env python3
"""GraphRAG CLI entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import GraphRAGPipeline
from src.visualization.graph_visualizer import GraphVisualizer


def cmd_build(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    pipeline = GraphRAGPipeline(config, run_id=args.run_id)
    data_dir = Path(args.data_dir) if args.data_dir else None
    pipeline.run_build(data_dir=data_dir)
    print(f"\nBuild complete. Log: {pipeline.logger.jsonl_path}")
    print(f"  Summary: {pipeline.logger.summary_path}")
    print(f"  Output: {config.output_path}")


def cmd_query(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    # load built index from disk
    pipeline = GraphRAGPipeline.load_from_disk(config)
    # get query engine
    engine = pipeline.get_query_engine(
        mode=args.mode,
        community_level=args.community_level,
    )
    response = engine.query(args.question)
    print("\n" + "=" * 60)
    print("Question:", args.question)
    print("Mode:", args.mode or config.query.default_mode)
    print("Community level:", args.community_level if args.community_level is not None else config.query.community_level)
    print("-" * 60)
    print("Answer:", str(response))
    print("=" * 60)


def cmd_steps(_: argparse.Namespace) -> None:
    print("GraphRAG Pipeline steps:\n")
    for step in GraphRAGPipeline.STEPS:
        print(f"  {step}")


def cmd_logs(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    logs_dir = config.logs_path
    if not logs_dir.exists():
        print("No logs found.")
        return

    log_files = sorted(logs_dir.glob("graphrag_*.jsonl"), reverse=True)
    if args.run_id:
        log_files = [logs_dir / f"graphrag_{args.run_id}.jsonl"]

    if not log_files:
        print("No logs found.")
        return

    log_file = log_files[0]
    print(f"Reading log: {log_file}\n")

    step_filter = args.step
    event_filter = args.event

    with open(log_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid log line in {log_file}", file=sys.stderr)
                continue
            if step_filter and record.get("step") != step_filter:
                continue
            if event_filter and record.get("event") != event_filter:
                continue
            print(json.dumps(record, ensure_ascii=False, indent=2))
            print("-" * 40)


def cmd_visualize(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    viz = GraphVisualizer(
        config.output_path,
        max_cluster_size=config.community.max_cluster_size,
    ).load()
    outputs = viz.visualize(
        level=args.level if args.level is not None else None,
        fmt=args.format,
    )
    print("\nVisualization generated:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    if args.format == "html":
        print("\nOpen the HTML files in a browser to explore (drag, zoom, hover for details).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LlamaIndex GraphRAG — full pipeline from chunking to community detection"
    )
    parser.add_argument(
        "--config", default=None, help="Path to config file (default: config.yaml)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Run the full build pipeline")
    p_build.add_argument("--data-dir", default=None, help="Data directory (default: data/)")
    p_build.add_argument("--run-id", default=None, help="Custom run_id for logs")
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="Query the built index")
    p_query.add_argument("question", help="Query question")
    p_query.add_argument(
        "--mode",
        choices=["local", "global", "auto"],
        default=None,
        help="Search mode: local (entity-centric), global (map-reduce over communities), auto (LLM routing)",
    )
    p_query.add_argument(
        "--community-level",
        type=int,
        default=None,
        help="Community hierarchy level for global search (0=most detailed)",
    )
    p_query.set_defaults(func=cmd_query)

    p_steps = sub.add_parser("steps", help="List pipeline steps")
    p_steps.set_defaults(func=cmd_steps)

    p_logs = sub.add_parser("logs", help="Query structured logs")
    p_logs.add_argument("--run-id", default=None, help="Filter by run_id")
    p_logs.add_argument(
        "--step",
        default=None,
        help="Filter by step: chunking / entity_extraction / community_detection, etc.",
    )
    p_logs.add_argument(
        "--event",
        default=None,
        help="Filter by event: chunk_extraction_done / community_detail, etc.",
    )
    p_logs.set_defaults(func=cmd_logs)

    p_viz = sub.add_parser("visualize", help="Visualize knowledge graph and communities")
    p_viz.add_argument(
        "--level",
        type=int,
        default=None,
        help="Render a single level (0=L0 leaf, 1=L1 mid, 2=L2 top); default: all levels",
    )
    p_viz.add_argument(
        "--format",
        choices=["html", "png"],
        default="html",
        help="Output format: html=interactive (PyVis), png=static (matplotlib)",
    )
    p_viz.set_defaults(func=cmd_visualize)

    args = parser.parse_args()
    try:
        args.func(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
