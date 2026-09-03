"""
Shared core functionality for academic path-flow visualization.

Extracted from academic_pathflow_visualizer.py and visualize_attack_graph.py
to eliminate code duplication and maintain consistent styling.
"""

from typing import Any, Dict, List, Optional, Tuple
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.pyplot as plt


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
    """Convert tactic name or ID to standardized TA format."""
    if tactic.startswith("TA") and tactic[2:6].isdigit():
        return tactic
    return TACTIC_ID_MAP.get(tactic, tactic)


class TrieNode:
    """Prefix-trie node for merging shared path prefixes."""
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


def build_prefix_trie(paths: List[Dict[str, Any]], top_k: int) -> Tuple[TrieNode, List[List[int]]]:
    """
    Build prefix trie from top-K paths and track node-id sequences per path for coloring.
    
    Args:
        paths: List of path dictionaries from predict JSON
        top_k: Number of top paths to include
        
    Returns:
        Tuple of (root TrieNode, list of node-id sequences per path)
    """
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
        track: List[int] = [cur.id]
        cum = 1.0
        for i, (tech, tac) in enumerate(zip(techs, tacs)):
            key = (tech, tac)
            if key not in cur.children:
                nid += 1
                child = TrieNode(nid, tech=tech, tac=tac, depth=cur.depth + 1, parent=cur)
                cur.children[key] = [child, 0.0, 0.0]
            else:
                child = cur.children[key][0]
            if i == 0:
                eprob = 1.0
                tac_prob = 1.0
            else:
                step_data = steps[i - 1] if (i - 1) < len(steps) else {}
                probs = step_data.get("probabilities", {})
                eprob = float(probs.get("combined", 1.0))
                tac_prob = float(probs.get("tactic_transition", eprob))
            cur.children[key][1] = max(cur.children[key][1], eprob)
            cur.children[key][2] = max(cur.children[key][2], tac_prob)
            cum *= eprob
            child.max_cumprob = max(child.max_cumprob, cum)
            cur = child
            track.append(cur.id)
        cur.is_leaf = True
        n_steps = len(steps) if steps else max(len(techs) - 1, 1)
        cur.leaf_meta = {
            "rank": p.get("rank"),
            "path_probability": p.get("path_probability", 0.0),
            "n_steps": n_steps,
            "path_confidence": p.get("path_confidence"),
        }
        path_tracks.append(track)

    for p in paths:
        add_path(p)
    return root, path_tracks


def compute_tidy_layout(root: TrieNode) -> Dict[int, Tuple[float, float]]:
    """
    Compute tidy tree layout with normalized coordinates.
    
    Args:
        root: Root TrieNode of the trie
        
    Returns:
        Dictionary mapping node IDs to (x, y) coordinates in [0, 1] range
    """
    pos: Dict[int, Tuple[int, float]] = {}
    y_counter = 0

    def layout(node: TrieNode) -> float:
        nonlocal y_counter
        if not node.children:
            y = y_counter
            y_counter += 1
            pos[node.id] = (node.depth, y)
            return y
        ys = [layout(entry[0]) for (_, _), entry in node.children.items()]
        y = sum(ys) / len(ys)
        pos[node.id] = (node.depth, y)
        return y

    layout(root)
    depths = [d for d, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    max_depth = max(depths) if depths else 1
    min_y, max_y = (min(ys), max(ys)) if ys else (0, 1)

    def norm(d: int, y: float) -> Tuple[float, float]:
        x = 0.05 + 0.90 * (d / (max_depth if max_depth else 1))
        yn = 0.10 + 0.80 * ((y - min_y) / (max_y - min_y)) if max_y != min_y else 0.5
        return x, yn

    return {nid: norm(d, y) for nid, (d, y) in pos.items()}


def draw_circle_node(
    ax,
    x: float,
    y: float,
    top: str,
    bottom: str,
    r: float = 0.035,
    edgecolor: str = "black",
    facecolor: str = "white",
) -> None:
    """
    Draw a two-line circular node.
    
    Args:
        ax: Matplotlib axes
        x, y: Node center coordinates
        top: Text for top line (technique ID)
        bottom: Text for bottom line (tactic ID)
        r: Circle radius
        edgecolor: Edge color
        facecolor: Fill color
    """
    circle = Circle((x, y), radius=r, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.6, zorder=10)
    ax.add_patch(circle)
    ax.text(x, y + r * 0.35, top, ha="center", va="center", fontsize=20, fontweight="bold", zorder=12)
    ax.text(x, y - r * 0.35, bottom, ha="center", va="center", fontsize=15, zorder=12)


def draw_arrow(
    ax,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    rad: float,
    lw: float,
    color: str,
) -> None:
    """
    Draw a curved arrow between two points.
    
    Args:
        ax: Matplotlib axes
        p1, p2: Start and end points
        rad: Arc radius (curvature)
        lw: Line width
        color: Arrow color
    """
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=lw,
        color=color,
        alpha=0.85,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arr)


def collect_nodes(root: TrieNode) -> Dict[int, TrieNode]:
    """
    Gather all TrieNode instances via DFS.
    
    Args:
        root: Root TrieNode
        
    Returns:
        Dictionary mapping node ID to TrieNode
    """
    nodes: Dict[int, TrieNode] = {}

    def dfs(n: TrieNode) -> None:
        nodes[n.id] = n
        for _, entry in n.children.items():
            dfs(entry[0])

    dfs(root)
    return nodes


def edge_anchor(
    p_from: Tuple[float, float],
    p_to: Tuple[float, float],
    r: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute edge start/end points offset from node centers by radius.
    
    Args:
        p_from, p_to: Node center coordinates
        r: Offset radius (typically node radius * 1.05)
        
    Returns:
        Tuple of (start_point, end_point)
    """
    x1, y1 = p_from
    x2, y2 = p_to
    dx, dy = x2 - x1, y2 - y1
    dist = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / dist, dy / dist
    start = (x1 + ux * r, y1 + uy * r)
    end = (x2 - ux * r, y2 - uy * r)
    return start, end


def edge_prob(parent: TrieNode, child: TrieNode) -> float:
    """Extract combined edge probability from parent to child."""
    for (_, _), entry in parent.children.items():
        if entry[0].id == child.id:
            return float(entry[1])
    return 1.0


def edge_tactic_prob(parent: TrieNode, child: TrieNode) -> float:
    """Extract tactic transition probability from parent to child."""
    for (_, _), entry in parent.children.items():
        if entry[0].id == child.id:
            return float(entry[2]) if len(entry) > 2 else float(entry[1])
    return 1.0
