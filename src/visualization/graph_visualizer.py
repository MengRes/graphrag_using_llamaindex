"""NetworkX + PyVis visualization for knowledge graphs and communities."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import networkx as nx
from llama_index.core.graph_stores.types import EntityNode

from src.graph_utils import build_entity_graph

# Community color palette (supports up to 50 communities)
COMMUNITY_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
    "#6699cc", "#99cc99", "#cc9999", "#cccc99", "#cc99cc",
    "#99cccc", "#ff9966", "#66ccff", "#cc66ff", "#99ff99",
]


def compute_communities(
    G: nx.Graph, max_cluster_size: int = 5
) -> list:
    """Run hierarchical Leiden and return cluster assignments."""
    from graspologic.partition import hierarchical_leiden

    if G.number_of_nodes() == 0:
        return []
    return hierarchical_leiden(G, max_cluster_size=max_cluster_size)


def load_cluster_assignments(store, json_path: Path | None = None) -> list | None:
    """Load persisted cluster assignments from store or graph_store.json."""
    if getattr(store, "cluster_assignments", None):
        return store.cluster_assignments
    if json_path and json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("cluster_assignments")
    return None


def assignments_to_clusters(assignments: list) -> list:
    return [
        SimpleNamespace(node=a["node"], cluster=a["cluster"], level=a["level"])
        for a in assignments
    ]


def community_map_at_level(clusters, level: int = 0) -> dict[str, int]:
    """Map node_id -> community_id at the given hierarchy level."""
    mapping: dict[str, int] = {}
    for item in clusters:
        if item.level == level:
            mapping[item.node] = item.cluster
    return mapping


def propagate_community_mapping(
    G: nx.Graph,
    clusters,
    level: int,
    l0_map: dict[str, int],
) -> tuple[dict[str, int], set[str]]:
    """
    Fill missing higher-level community assignments by propagating from explicit
    Leiden anchors through L0 co-membership and graph connectivity.

    Returns (full_mapping, explicit_node_ids).
    """
    explicit = community_map_at_level(clusters, level=level)
    if level == 0:
        return explicit, set(explicit.keys())

    from collections import Counter, defaultdict, deque

    by_l0: dict[int, list[str]] = defaultdict(list)
    for node, cluster in l0_map.items():
        by_l0[cluster].append(node)

    result = dict(explicit)
    explicit_nodes = set(explicit.keys())

    for members in by_l0.values():
        anchors = [m for m in members if m in result]
        if not anchors:
            continue
        winner = Counter(result[m] for m in anchors).most_common(1)[0][0]
        for member in members:
            if member not in result:
                result[member] = winner

    for start in (n for n in G.nodes if n not in result):
        queue = deque([start])
        visited = {start}
        inherited: int | None = None
        while queue and inherited is None:
            node = queue.popleft()
            for neighbor in G.neighbors(node):
                if neighbor in result:
                    inherited = result[neighbor]
                    break
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if inherited is not None:
            result[start] = inherited

    if result:
        communities = sorted(set(result.values()))
        for node in G.nodes:
            if node not in result:
                result[node] = communities[l0_map[node] % len(communities)]

    return result, explicit_nodes


def _node_color(community_id: Optional[int]) -> str:
    if community_id is None:
        return "#cccccc"
    return COMMUNITY_COLORS[community_id % len(COMMUNITY_COLORS)]


def render_pyvis_html(
    G: nx.Graph,
    output_path: Path,
    title: str,
    community_mapping: Optional[dict[str, int]] = None,
    explicit_community_nodes: Optional[set[str]] = None,
    height: str = "800px",
    width: str = "100%",
) -> Path:
    """Render an interactive HTML network graph."""
    from pyvis.network import Network

    net = Network(
        height=height,
        width=width,
        directed=False,
        notebook=False,
        bgcolor="#ffffff",
        font_color="#333333",
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.04,
    )

    for node_id, data in G.nodes(data=True):
        label = data.get("label", node_id)
        entity_type = data.get("entity_type", "entity")
        cid = community_mapping.get(node_id) if community_mapping else None
        color = _node_color(cid)

        if community_mapping is not None and cid is not None:
            inferred = (
                explicit_community_nodes is not None
                and node_id not in explicit_community_nodes
            )
            suffix = " (inferred)" if inferred else ""
            node_title = f"{label}\nType: {entity_type}\nCommunity: {cid}{suffix}"
        else:
            node_title = f"{label}\nType: {entity_type}"

        display_label = label if len(label) <= 20 else label[:18] + "…"
        net.add_node(
            node_id,
            label=display_label,
            title=node_title,
            color=color,
            size=18 if community_mapping else 14,
        )

    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        desc = data.get("description", "")
        net.add_edge(u, v, title=f"{rel}: {desc}", label=rel[:15] if rel else "")

    net.set_options(
        """
    {
      "nodes": {"font": {"size": 12}},
      "edges": {
        "font": {"size": 9, "align": "middle"},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "stabilization": {"iterations": 150}
      }
    }
    """
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = net.generate_html(notebook=False)
    header = f"<h2 style='font-family:sans-serif;padding:12px'>{title}</h2>\n"
    if community_mapping:
        n_comm = len(set(community_mapping.values()))
        n_inferred = 0
        if explicit_community_nodes is not None:
            n_inferred = sum(
                1 for n in G.nodes if n not in explicit_community_nodes
            )
        header += (
            f"<p style='font-family:sans-serif;padding:0 12px;color:#666'>"
            f"Nodes {G.number_of_nodes()} · Edges {G.number_of_edges()} · "
            f"Communities {n_comm} (color-coded)"
        )
        if n_inferred:
            header += (
                f" · {n_inferred} nodes with inferred community "
                f"(hover for details)"
            )
        header += "</p>\n"
    else:
        header += (
            f"<p style='font-family:sans-serif;padding:0 12px;color:#666'>"
            f"Nodes {G.number_of_nodes()} · Edges {G.number_of_edges()}</p>\n"
        )
    html = html.replace("<body>", f"<body>\n{header}", 1)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def render_matplotlib_png(
    G: nx.Graph,
    output_path: Path,
    title: str,
    community_mapping: Optional[dict[str, int]] = None,
) -> Path:
    """Render a static PNG (NetworkX + matplotlib)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, k=1.5, seed=42, iterations=50)

    if community_mapping:
        node_colors = [_node_color(community_mapping.get(n)) for n in G.nodes()]
    else:
        node_colors = "#6699cc"

    labels = {n: (G.nodes[n].get("label", n)[:12]) for n in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


class GraphVisualizer:
    """Load graph from built output and generate visualizations."""

    def __init__(self, output_dir: Path, max_cluster_size: int = 5):
        self.output_dir = Path(output_dir)
        self.max_cluster_size = max_cluster_size
        self._graph_store = None

    def load(self):
        pkl = self.output_dir / "graph_store.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"Not found: {pkl}. Run build first.")
        with open(pkl, "rb") as f:
            self._graph_store = pickle.load(f)
        return self

    @property
    def entity_graph(self) -> nx.Graph:
        if not self._graph_store:
            raise RuntimeError("Call load() first")
        return build_entity_graph(self._graph_store.graph)

    def visualize(
        self,
        viz_dir: Optional[Path] = None,
        level: Optional[int] = None,
        fmt: str = "html",
    ) -> dict[str, Path]:
        """
        Generate knowledge graph and community-colored views.

        Args:
            level: Single hierarchy level (0/1/2); None renders all available levels.

        Returns:
            Dict of output name -> file path.
        """
        viz_dir = viz_dir or (self.output_dir / "viz")
        G = self.entity_graph
        json_path = self.output_dir / "graph_store.json"
        assignments = load_cluster_assignments(self._graph_store, json_path)
        if assignments:
            clusters = assignments_to_clusters(assignments)
        else:
            clusters = compute_communities(G, self.max_cluster_size)
        levels_available = sorted({c.level for c in clusters})
        l0_map = community_map_at_level(clusters, level=0)

        outputs: dict[str, Path] = {}
        ext = "html" if fmt == "html" else "png"
        render_fn = render_pyvis_html if fmt == "html" else render_matplotlib_png

        kg_path = viz_dir / f"knowledge_graph.{ext}"
        render_fn(G, kg_path, "Knowledge Graph (Entity-Relation Extraction)")
        outputs["knowledge_graph"] = kg_path

        levels_to_render = [level] if level is not None else levels_available
        level_labels = {0: "L0 leaf communities", 1: "L1 mid-level", 2: "L2 top-level"}
        propagated_maps: dict[int, dict[str, int]] = {}
        explicit_by_level: dict[int, set[str]] = {}

        for lv in levels_to_render:
            if lv not in levels_available:
                continue
            if lv == 0:
                comm_map = l0_map
                explicit_nodes = set(l0_map.keys())
            else:
                comm_map, explicit_nodes = propagate_community_mapping(
                    G, clusters, level=lv, l0_map=l0_map
                )
            propagated_maps[lv] = comm_map
            explicit_by_level[lv] = explicit_nodes

            n_comm = len(set(comm_map.values()))
            label = level_labels.get(lv, f"Level {lv}")
            comm_path = viz_dir / f"community_level{lv}.{ext}"
            if fmt == "html":
                render_pyvis_html(
                    G,
                    comm_path,
                    f"Community Detection — {label} ({n_comm} communities)",
                    community_mapping=comm_map,
                    explicit_community_nodes=explicit_nodes,
                )
            else:
                render_matplotlib_png(
                    G,
                    comm_path,
                    f"Community Detection — {label} ({n_comm} communities)",
                    community_mapping=comm_map,
                )
            outputs[f"community_level{lv}"] = comm_path

        assignment_path = viz_dir / "community_assignments.json"
        assignment_data = {
            "levels_available": levels_available,
            "max_cluster_size": self.max_cluster_size,
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "community_counts": {
                str(lv): len(set(propagated_maps[lv].values()))
                for lv in propagated_maps
            },
            "explicit_node_counts": {
                str(lv): len(explicit_by_level[lv]) for lv in explicit_by_level
            },
            "by_level": {
                str(lv): propagated_maps[lv] for lv in propagated_maps
            },
            "explicit_by_level": {
                str(lv): sorted(explicit_by_level[lv]) for lv in explicit_by_level
            },
        }
        with open(assignment_path, "w", encoding="utf-8") as f:
            json.dump(assignment_data, f, ensure_ascii=False, indent=2)
        outputs["assignments"] = assignment_path

        return outputs
