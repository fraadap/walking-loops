"""
Metric computation module for walking-loop evaluation.

All public functions accept raw Python/NumPy data and return plain dicts.
No SB3/Gymnasium imports here — keep this module importable from anywhere.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------------
# Public constants (shared between train / test / optuna pipelines)
# ---------------------------------------------------------------------------

EVAL_SEED = 42
EVAL_CITIES = [
    "Florence, Italy",
    "Prague, Czech Republic",
    "Copenhagen, Denmark",
    "Bruges, Belgium",
    "Salzburg, Austria",
]
N_EVAL_PER_CITY = 20  # 5 cities × 20 = 100 deterministic episodes


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def compute_loop_metrics(
    G,
    path_nodes: list,
    target_duration: float,
    actual_duration: float,
    radius_meters: float = 3000.0,
    episode_pois: list | None = None,
    user_vec: np.ndarray | None = None,
) -> dict:
    """
    Compute a comprehensive metric dict for one completed walking loop.

    Parameters
    ----------
    G : networkx.DiGraph  (OSMnx street graph)
    path_nodes            : ordered list of node IDs forming the loop
    target_duration       : requested duration in minutes
    actual_duration       : realised duration in minutes
    radius_meters         : env radius (used to normalise compactness)
    episode_pois          : list of POI node IDs active in this episode
    user_vec              : 4-dim preference vector in [-1, 1]^4 (optional)

    Returns
    -------
    dict with keys grouped into: temporal, spatial, topological, poi, personal.
    """
    coords = np.array(
        [[G.nodes[n]["x"], G.nodes[n]["y"]] for n in path_nodes],
        dtype=np.float64,
    )

    m: dict = {}

    # ------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------
    td = float(max(target_duration, 1e-6))
    m["pct_error"] = float(abs(actual_duration - target_duration) / td)
    m["signed_error"] = float((actual_duration - target_duration) / td)
    m["actual_duration"] = float(actual_duration)
    m["target_duration"] = float(target_duration)
    m["target_bucket"] = _bucket(target_duration)
    m["success_10"] = float(m["pct_error"] <= 0.10)
    m["success_20"] = float(m["pct_error"] <= 0.20)

    # ------------------------------------------------------------------
    # Spatial / Shape
    # ------------------------------------------------------------------
    m["circularity"] = _shoelace_circularity(coords)
    m["max_dist_from_home_m"] = _max_dist_from_home_m(coords)
    m["compactness"] = float(min(1.0, m["max_dist_from_home_m"] / max(radius_meters, 1.0)))
    m["convex_hull_area_sqm"] = _convex_hull_area_sqm(coords)
    m["mean_turning_angle_rad"] = _mean_turning_angle_rad(coords)

    # ------------------------------------------------------------------
    # Topological
    # ------------------------------------------------------------------
    if len(path_nodes) > 1:
        edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        unique_edges = {tuple(sorted(e)) for e in edges}
        m["edge_revisit_ratio"] = float(1.0 - len(unique_edges) / max(1, len(edges)))
        unique_nodes = set(path_nodes)
        m["node_revisit_ratio"] = float(1.0 - len(unique_nodes) / max(1, len(path_nodes)))
        m["path_length"] = len(path_nodes)

        seg_lengths = []
        for u, v in edges:
            if v in G[u]:
                edge_data = G[u][v]
                # MultiDiGraph: G[u][v] is {key: attr_dict}; DiGraph: G[u][v] is attr_dict
                if isinstance(next(iter(edge_data.values()), None), dict):
                    length = min(d.get("length", 0.0) for d in edge_data.values())
                else:
                    length = edge_data.get("length", 0.0)
                seg_lengths.append(float(length))
        if seg_lengths:
            m["mean_segment_len_m"] = float(np.mean(seg_lengths))
            m["micro_segments_pct"] = float(
                sum(1 for s in seg_lengths if s < 30) / len(seg_lengths)
            )
        else:
            m["mean_segment_len_m"] = 0.0
            m["micro_segments_pct"] = 0.0
    else:
        m["edge_revisit_ratio"] = 0.0
        m["node_revisit_ratio"] = 0.0
        m["path_length"] = 1
        m["mean_segment_len_m"] = 0.0
        m["micro_segments_pct"] = 0.0

    # ------------------------------------------------------------------
    # POI
    # ------------------------------------------------------------------
    if episode_pois is not None:
        visited_pois = set(path_nodes) & set(episode_pois)
        m["poi_count"] = len(visited_pois)
        m["poi_density_per_min"] = float(len(visited_pois) / max(1.0, actual_duration))
    else:
        m["poi_count"] = 0
        m["poi_density_per_min"] = 0.0

    # ------------------------------------------------------------------
    # Personalisation — preference alignment (optional)
    # ------------------------------------------------------------------
    if user_vec is not None:
        uv = np.asarray(user_vec, dtype=np.float64)
        features = np.array(
            [
                m["circularity"],
                m["compactness"],
                1.0 - m["edge_revisit_ratio"],
                float(min(1.0, m["poi_density_per_min"] / 0.5)),
            ],
            dtype=np.float64,
        )
        dot = float(np.dot(uv, features))
        norm = float(np.linalg.norm(uv))
        m["preference_alignment"] = dot
        m["preference_alignment_normalized"] = dot / (norm + 1e-9)
    else:
        m["preference_alignment"] = float("nan")
        m["preference_alignment_normalized"] = float("nan")

    return m


def make_deterministic_test_set(
    cities: list[str] = EVAL_CITIES,
    n_per_city: int = N_EVAL_PER_CITY,
    seed: int = EVAL_SEED,
) -> list[dict]:
    """
    Generate a fixed, reproducible evaluation set.

    Returns a list of dicts: {'city': str, 'target_duration': float}.
    Same seed → same list every call — essential for fair Optuna comparison.
    """
    rng = np.random.RandomState(seed)
    cases = []
    for city in cities:
        targets = rng.uniform(10.0, 120.0, size=n_per_city)
        for t in targets:
            cases.append({"city": city, "target_duration": float(t)})
    # Deterministic shuffle so cities interleave (avoids early-stopping bias in pruning)
    rng.shuffle(cases)
    return cases


# ---------------------------------------------------------------------------
# Private geometry helpers
# ---------------------------------------------------------------------------

def _shoelace_circularity(coords: np.ndarray) -> float:
    """Ratio of loop area to ideal-circle area for the same perimeter. In [0, 1]."""
    if len(coords) < 4:
        return 0.0
    mean_lat = float(np.mean(coords[:, 1]))
    cos_lat = np.cos(np.radians(mean_lat))
    xs = coords[:, 0] * cos_lat  # stretch lon to approx isometric
    ys = coords[:, 1]

    area_deg2 = 0.5 * abs(
        float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    )
    area_sqm = area_deg2 * (111_111.0 ** 2)

    dx = np.diff(xs) * 111_111.0
    dy = np.diff(ys) * 111_111.0
    perim = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))

    if perim < 1.0:
        return 0.0
    ideal_area = (perim ** 2) / (4 * np.pi)
    return float(min(1.0, area_sqm / max(1.0, ideal_area)))


def _convex_hull_area_sqm(coords: np.ndarray) -> float:
    """Convex hull area of the path in m²."""
    if len(coords) < 4:
        return 0.0
    try:
        mean_lat = float(np.mean(coords[:, 1]))
        cos_lat = np.cos(np.radians(mean_lat))
        pts = np.column_stack([
            coords[:, 0] * cos_lat * 111_111.0,
            coords[:, 1] * 111_111.0,
        ])
        # ConvexHull.volume is area in 2-D
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


def _mean_turning_angle_rad(coords: np.ndarray) -> float:
    """Mean absolute turning angle in radians (0 = straight, π = U-turn)."""
    if len(coords) < 3:
        return 0.0
    vecs = np.diff(coords, axis=0).astype(np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs_n = vecs / norms
    dots = np.clip(np.einsum("ij,ij->i", vecs_n[:-1], vecs_n[1:]), -1.0, 1.0)
    return float(np.mean(np.arccos(dots)))


def _max_dist_from_home_m(coords: np.ndarray) -> float:
    """Maximum Euclidean distance from the first node (home) in metres (approx)."""
    if len(coords) < 2:
        return 0.0
    home = coords[0]
    mean_lat = float(np.mean(coords[:, 1]))
    cos_lat = np.cos(np.radians(mean_lat))
    dx_m = (coords[:, 0] - home[0]) * cos_lat * 111_111.0
    dy_m = (coords[:, 1] - home[1]) * 111_111.0
    return float(np.max(np.sqrt(dx_m ** 2 + dy_m ** 2)))


def _bucket(td: float) -> str:
    if td <= 30:
        return "10-30"
    if td <= 60:
        return "30-60"
    if td <= 90:
        return "60-90"
    return "90-120"
