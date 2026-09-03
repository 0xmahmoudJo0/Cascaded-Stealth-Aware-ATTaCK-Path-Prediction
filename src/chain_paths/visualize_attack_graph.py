"""
Academic Probabilistic State Transition Graph (Path-flow with tactic branching)

Nodes: circular split labels (top=technique, bottom=tactic), per-path colored circles, neutral thin curved arrows.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt

from .core_pathflow import (
    tactic_to_id,
    build_prefix_trie,
    compute_tidy_layout,
    draw_circle_node,
    draw_arrow,
    collect_nodes,
    edge_anchor,
    edge_prob,
    edge_tactic_prob,
)


def render_academic_pathflow(
    json_path: Path,
    output_png: Path,
    top_k_paths: int = 6,
    node_r: float = 0.052,
    dpi: int = 300,
    show_probabilities: bool = True,
):
    """
    Render academic path-flow graph from predict JSON.
    
    Args:
        json_path: Path to predict JSON file
        output_png: Output image path
        top_k_paths: Number of top paths to include
        node_r: Node circle radius
        dpi: Image resolution
        show_probabilities: Whether to show leaf node probability labels
    """
    data = json.loads(Path(json_path).read_text())
    paths = data.get("paths", [])
    root, path_tracks = build_prefix_trie(paths, top_k=top_k_paths)
    pos = compute_tidy_layout(root)
    nodes_by_id = collect_nodes(root)

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    node_face: Dict[int, str] = {}
    for p_idx, track in enumerate(path_tracks):
        c = palette[p_idx % len(palette)]
        for nid in track[1:]:
            node_face.setdefault(nid, c)

    fig = plt.figure(figsize=(24, 11))
    ax = plt.gca()
    ax.set_xlim(-0.02, 1.20)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    branching_edge = set()
    for _, n in nodes_by_id.items():
        if len(n.children) > 1:
            for _, entry in n.children.items():
                branching_edge.add((n.id, entry[0].id))

    arrow_color = "#555555"
    label_drawn = set()
    for track in path_tracks:
        for i in range(len(track) - 1):
            pid = track[i]
            cid = track[i + 1]
            parent = nodes_by_id[pid]
            child = nodes_by_id[cid]
            x1, y1 = pos[pid]
            x2, y2 = pos[cid]
            start, end = edge_anchor((x1, y1), (x2, y2), node_r * 1.05)
            eprob = edge_prob(parent, child)
            lw = 0.4 + 1.4 * max(0.0, min(1.0, eprob))
            rad = 0.0 if abs(y2 - y1) < 0.08 else (0.12 if y2 > y1 else -0.12)

            draw_arrow(ax, start, end, rad=rad, lw=lw, color=arrow_color)



    for _, n in nodes_by_id.items():
        x, y = pos[n.id]
        if n.tech == "START":
            draw_circle_node(ax, x, y, "Start", "START", r=node_r, edgecolor="black", facecolor="#E8F6F3")
        else:
            face = node_face.get(n.id, "#FFFFFF")
            draw_circle_node(ax, x, y, n.tech, tactic_to_id(n.tac), r=node_r, edgecolor="black", facecolor=face)
            if show_probabilities and n.is_leaf and n.leaf_meta:
                path_conf = n.leaf_meta.get('path_confidence')
                if path_conf is not None:
                    conf_pct = path_conf
                else:
                    path_prob = n.leaf_meta['path_probability']
                    n_steps = n.leaf_meta.get('n_steps', 1)
                    geo_mean = path_prob ** (1.0 / max(n_steps, 1)) if path_prob > 0 else 0.0
                    conf_pct = geo_mean * 100
                ax.text(
                    x,
                    y - node_r - 0.028,
                    f"rank {n.leaf_meta['rank']} | Probability={conf_pct:.1f}%",
                    ha="center",
                    va="top",
                    fontsize=11,
                )

    ax.set_title(
        "Log-Linear Fusion with Tactic-Branching Beam Search: Predicted Attack Paths from Seed T1190\n"
        "Nodes: [Technique ID | Tactic ID]  —  Edge labels: $P(\\tau_{t+1} \\mid \\tau_t)$  —  Ranked by joint path probability",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_from_cli(
    input_path: Path,
    output_path: Path,
    max_paths: Optional[int] = None,
    show_probabilities: bool = True,
    output_format: str = 'png',
    dpi: int = 300,
):
    """Render academic path-flow graph from predict JSON."""
    if output_path.is_dir():
        output_file = output_path / f"academic_pathflow.{output_format}"
    else:
        output_file = output_path
        if output_file.suffix.lower() != f".{output_format}":
            output_file = output_file.with_suffix(f".{output_format}")

    top_k = max_paths if max_paths else 6

    render_academic_pathflow(
        json_path=Path(input_path),
        output_png=output_file,
        top_k_paths=top_k,
        dpi=dpi,
        show_probabilities=show_probabilities,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualize_attack_graph.py <input.json> [output.png] [top_k]")
        sys.exit(1)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.parent / "academic_pathflow.png"
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    render_academic_pathflow(inp, out, top_k_paths=topk)

    """
    Academic Probabilistic State Transition Graph (Path-flow with tactic branching)

    Nodes: circular split labels (top=tactic, bottom=technique), per-path colored ovals, neutral thin arrows.
    """

    import json
    from pathlib import Path
    from typing import Dict, Any, Optional, List, Tuple

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Circle


    TACTIC_ID_MAP = {
        "reconnaissance": "TA0043",
        "resource-development": "TA0042",
        "initial-access": "TA0001",
        "execution": "TA0002",
        "persistence": "TA0003",
        "privilege-escalation": "TA0004",
        "defense-evasion": "TA0005",
        "credential-access": "TA0006",
        "discovery": "TA0007",
        "lateral-movement": "TA0008",
        "collection": "TA0009",
        "command-and-control": "TA0011",
        "exfiltration": "TA0010",
        "impact": "TA0040",
    }


    def tactic_to_id(tactic: str) -> str:
        if tactic.startswith("TA") and tactic[2:6].isdigit():
            return tactic
        return TACTIC_ID_MAP.get(tactic, tactic)


    class TrieNode:
        __slots__ = ("id", "tech", "tac", "depth", "children", "parent", "max_cumprob", "is_leaf", "leaf_meta")

        def __init__(self, nid: int, tech: str, tac: str, depth: int, parent: Optional["TrieNode"] = None):
            self.id = nid
            self.tech = tech
            self.tac = tac
            self.depth = depth
            self.children: Dict[Tuple[str, str], List[Any]] = {}
            self.parent = parent
            self.max_cumprob = 0.0
            self.is_leaf = False
            self.leaf_meta = None


    def build_prefix_trie(paths: List[Dict[str, Any]], top_k: int = 6) -> Tuple[TrieNode, List[List[int]]]:
        paths = sorted(paths, key=lambda p: p.get("path_probability", 0.0), reverse=True)[:top_k]
        nid = 0
        root = TrieNode(nid, tech="START", tac="Start", depth=0)
        path_tracks: List[List[int]] = []

        def add_path(p: Dict[str, Any]):
            nonlocal nid
            techs = p["techniques"]
            tacs = p["tactic_sequence"]
            steps = p.get("steps", [])
            cur = root
            track = [cur.id]
            cum = 1.0
            for i, (tech, tac) in enumerate(zip(techs, tacs)):
                key = (tech, tac)
                if key not in cur.children:
                    nid += 1
                    child = TrieNode(nid, tech=tech, tac=tac, depth=cur.depth + 1, parent=cur)
                    cur.children[key] = [child, 0.0]
                else:
                    child = cur.children[key][0]
                if i == 0:
                    eprob = 1.0
                else:
                    eprob = steps[i - 1]["probabilities"].get("combined", 1.0) if (i - 1) < len(steps) else 1.0
                eprob = float(eprob)
                cur.children[key][1] = max(cur.children[key][1], eprob)
                cum *= eprob
                child.max_cumprob = max(child.max_cumprob, cum)
                cur = child
                track.append(cur.id)
            cur.is_leaf = True
            cur.leaf_meta = {"rank": p.get("rank"), "path_probability": p.get("path_probability", 0.0)}
            path_tracks.append(track)

        for p in paths:
            add_path(p)

        return root, path_tracks


    def compute_tidy_layout(root: TrieNode):
        pos = {}
        y_counter = 0

        def layout(node: TrieNode):
            nonlocal y_counter
            if not node.children:
                y = y_counter
                y_counter += 1
                pos[node.id] = (node.depth, y)
                return y
            ys = [layout(child) for (_, _), (child, _) in node.children.items()]
            y = sum(ys) / len(ys)
            pos[node.id] = (node.depth, y)
            return y

        layout(root)
        depths = [d for d, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        max_depth = max(depths) if depths else 1
        min_y, max_y = (min(ys), max(ys)) if ys else (0, 1)

        def norm(d, y):
            x = 0.05 + 0.90 * (d / (max_depth if max_depth else 1))
            yn = 0.10 + 0.80 * ((y - min_y) / (max_y - min_y)) if max_y != min_y else 0.5
            return x, yn

        return {nid: norm(d, y) for nid, (d, y) in pos.items()}


    def draw_circle_node(ax, x, y, top, bottom, r=0.03, fc="white", ec="black", lw=1.2, fs_top=8, fs_bot=7):
        circle = Circle((x, y), radius=r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y + r * 0.35, top, ha="center", va="center", fontsize=fs_top, fontweight="bold", zorder=12)
        ax.text(x, y - r * 0.35, bottom, ha="center", va="center", fontsize=fs_bot, zorder=12)


    def draw_arrow(ax, p1, p2, rad=0.0, lw=1.0, color="black", alpha=0.8, z=2):
        arr = FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
        )
        ax.add_patch(arr)


    def collect_nodes(root: TrieNode) -> Dict[int, TrieNode]:
        nodes: Dict[int, TrieNode] = {}

        def dfs(n: TrieNode):
            nodes[n.id] = n
            for _, (c, _) in n.children.items():
                dfs(c)

        dfs(root)
        return nodes


    def render_academic_pathflow(
        json_path: Path,
        output_png: Path,
        top_k_paths: int = 6,
        node_r: float = 0.03,
        dpi: int = 300,
    ):
        data = json.loads(Path(json_path).read_text())
        paths = data.get("paths", [])
        root, path_tracks = build_prefix_trie(paths, top_k=top_k_paths)
        pos = compute_tidy_layout(root)
        nodes_by_id = collect_nodes(root)

        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        node_face: Dict[int, str] = {}
        for p_idx, track in enumerate(path_tracks):
            c = palette[p_idx % len(palette)]
            for nid in track[1:]:
                node_face.setdefault(nid, c)

        fig = plt.figure(figsize=(18, 6))
        ax = plt.gca()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        def edge_anchor(p_from: Tuple[float, float], p_to: Tuple[float, float], r: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
            x1, y1 = p_from
            x2, y2 = p_to
            dx, dy = x2 - x1, y2 - y1
            dist = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            ux, uy = dx / dist, dy / dist
            start = (x1 + ux * r, y1 + uy * r)
            end = (x2 - ux * r, y2 - uy * r)
            return start, end

        branching_edge = set()
        for _, n in nodes_by_id.items():
            if len(n.children) > 1:
                for _, (child, _) in n.children.items():
                    branching_edge.add((n.id, child.id))

        def edge_prob(parent: TrieNode, child: TrieNode) -> float:
            for (_, _), (c, prob) in parent.children.items():
                if c.id == child.id:
                    return float(prob)
            return 1.0

        arrow_color = "#555555"
        label_drawn = set()
        for track in path_tracks:
            for i in range(len(track) - 1):
                pid = track[i]
                cid = track[i + 1]
                parent = nodes_by_id[pid]
                child = nodes_by_id[cid]
                x1, y1 = pos[pid]
                x2, y2 = pos[cid]
                start, end = edge_anchor((x1, y1), (x2, y2), node_r * 1.05)
                eprob = edge_prob(parent, child)
                lw = 0.35 + 1.2 * max(0.0, min(1.0, eprob))
                rad = 0.14 if abs(y2 - y1) < 0.08 else (0.20 if y2 > y1 else -0.20)

                draw_arrow(ax, start, end, rad=rad, lw=lw, color=arrow_color, alpha=0.85, z=2)

                if (pid, cid) in branching_edge and (pid, cid) not in label_drawn:
                    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
                    ax.text(
                        mx,
                        my + 0.03,
                        f"{tactic_to_id(child.tac)}\n{eprob:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=arrow_color, alpha=0.9, linewidth=0.8),
                        zorder=5,
                    )
                    label_drawn.add((pid, cid))

        for _, n in nodes_by_id.items():
            x, y = pos[n.id]
            if n.tech == "START":
                draw_circle_node(ax, x, y, "START", "Start", r=node_r, fc="#E8F6F3", ec="black", lw=1.4, fs_top=9, fs_bot=8)
            else:
                face = node_face.get(n.id, "#FFFFFF")
                draw_circle_node(ax, x, y, tactic_to_id(n.tac), n.tech, r=node_r, fc=face, ec="black", lw=1.2, fs_top=8, fs_bot=7)
                if n.is_leaf and n.leaf_meta:
                    ax.text(
                        x,
                        y - node_r - 0.028,
                        f"rank {n.leaf_meta['rank']} | P={n.leaf_meta['path_probability']:.3e}",
                        ha="center",
                        va="top",
                        fontsize=7,
                    )

        ax.set_title(
            "Probabilistic State Transition Graph (Academic Path-flow with Tactic Branching)\n"
        "Nodes: [Technique (top) | Tactic (bottom)] — Branches represent full Top-K predicted paths",
            pad=14,
        )

        plt.tight_layout()
        plt.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)


    def visualize_from_cli(
        input_path: Path,
        output_path: Path,
        max_paths: Optional[int] = None,
        show_probabilities: bool = True,
        output_format: str = 'png',
        dpi: int = 300,
    ):
        if output_path.is_dir():
            output_file = output_path / f"academic_pathflow.{output_format}"
        else:
            output_file = output_path
            if output_file.suffix.lower() != f".{output_format}":
                output_file = output_file.with_suffix(f".{output_format}")

        top_k = max_paths if max_paths else 6

        render_academic_pathflow(
            json_path=Path(input_path),
            output_png=output_file,
            top_k_paths=top_k,
            dpi=dpi,
        )


    if __name__ == "__main__":
        import sys
        if len(sys.argv) < 2:
            print("Usage: python visualize_attack_graph.py <input.json> [output.png] [top_k]")
            sys.exit(1)
        inp = Path(sys.argv[1])
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.parent / "academic_pathflow.png"
        topk = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        render_academic_pathflow(inp, out, top_k_paths=topk)
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    render_academic_pathflow(inp, out, top_k_paths=topk)
