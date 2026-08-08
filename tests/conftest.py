"""
Shared fixtures for v2 tests.

Uses the same synthetic 4×4 grid as v1 conftest, injected via mock
so no OSM download is required.
"""
from __future__ import annotations

import copy
import os
import sys
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_grid_graph(rows: int = 4, cols: int = 4, edge_len: float = 111.0) -> nx.DiGraph:
    G = nx.DiGraph()
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            G.add_node(nid, x=c * 0.001, y=r * 0.001, osmid=nid)
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            if c < cols - 1:
                G.add_edge(nid, nid + 1, length=edge_len, key=0)
                G.add_edge(nid + 1, nid, length=edge_len, key=0)
            if r < rows - 1:
                G.add_edge(nid, nid + cols, length=edge_len, key=0)
                G.add_edge(nid + cols, nid, length=edge_len, key=0)
    return G


def _precompute_paths(G, key_nodes):
    paths, path_lengths = {}, {}
    for u in key_nodes:
        paths[u], path_lengths[u] = {}, {}
        lengths, all_paths = nx.single_source_dijkstra(G, u, weight="length")
        for v in key_nodes:
            paths[u][v] = all_paths.get(v, [])
            path_lengths[u][v] = lengths.get(v, float("inf"))
    return paths, path_lengths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def grid_graph():
    return _build_grid_graph()


@pytest.fixture(scope="session")
def city_data_v2(grid_graph):
    home = 0
    pois = list(range(1, 10))
    key_nodes = [home] + pois
    paths, path_lengths = _precompute_paths(grid_graph, key_nodes)
    return {"G": grid_graph, "home_node": home, "pois": pois,
            "paths": paths, "path_lengths": path_lengths}


@pytest.fixture()
def mock_env_v2(city_data_v2):
    from walking_loops.env import WalkingEnvV2
    from walking_loops.config import EnvConfig

    cfg = EnvConfig(num_pois=len(city_data_v2["pois"]), radius_meters=1000)
    G = city_data_v2["G"]
    home = city_data_v2["home_node"]
    pois = city_data_v2["pois"]
    paths = city_data_v2["paths"]
    path_lengths = city_data_v2["path_lengths"]

    with patch("walking_loops.env.get_or_create_graph") as mock_goc:
        mock_goc.return_value = (
            G, home, pois,
            copy.deepcopy(paths),
            copy.deepcopy(path_lengths),
        )
        env = WalkingEnvV2(places=["TestCity"], cfg=cfg)
    return env
