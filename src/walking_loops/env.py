"""
WalkingEnvV2 — Gymnasium environment for walking-loop planning.

Key improvements over v1:
  - Balanced reward: new_node=+0.3, revisit=-0.3  (was 0.01 / -1.0 → 100:1 imbalance)
  - Circularity bonus always proportional to timing quality (no hard gate)
  - Dense temporal shaping at every step (potential-based, Ng et al. 1999)
  - Curriculum API: set_target_range(min_t, max_t) for external control
  - max_steps increased 50→60 for longer episodes
  - All v1 masking fixes retained (step_count < 1 gate, feasibility feature)
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

# Allow importing from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from walking_loops.graph_utils import get_or_create_graph
from walking_loops.config import EnvConfig


class WalkingEnvV2(gym.Env):
    """
    Sequential POI-selection environment for walking loops.

    Observation (flat):
      [5 scalars | 2 home_vec | 6 breadcrumbs | N × 5 POI features]
      scalars: target_norm, current_norm, t_home_norm, delta_norm, urgency_norm
      POI:     visited, dx, dy, t_to_poi_norm, round_trip_feasibility

    Action:
      0 … N-1 : visit episode_pois[action]
      N       : go home (terminates episode)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        places: list[str] | str,
        cfg: EnvConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or EnvConfig()

        if isinstance(places, str):
            places = [places]
        self.places = places

        c = self.cfg
        self.WALKING_SPEED_M_MIN: float = c.walking_speed_kmh * 1000.0 / 60.0
        self.MAX_TARGET_DURATION: float = c.max_target_duration
        self.MAX_COORD_DIFF: float = c.max_coord_diff
        self.max_steps: int = c.max_steps
        self.num_pois: int = c.num_pois

        # Curriculum: externally adjustable target range
        self._min_target: float = c.initial_min_target
        self._max_target: float = c.initial_max_target

        print("WalkingEnvV2: loading cities …")
        self.city_data: dict = {}
        for place in self.places:
            G, home, pois, paths, path_lengths = get_or_create_graph(
                place, c.num_pois, c.radius_meters
            )
            self.city_data[place] = {
                "G": G,
                "home_node": home,
                "pois": pois,
                "paths": paths,
                "path_lengths": path_lengths,
            }

        self._set_active_city(self.places[0])

        self.action_space = spaces.Discrete(self.num_pois + 1)
        obs_dim = 5 + 2 + 6 + self.num_pois * 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # EWMA of per-city error for prioritised sampling
        self.city_error_ewma: dict[str, float] = {p: 1.0 for p in self.places}

    # ------------------------------------------------------------------
    # Curriculum API
    # ------------------------------------------------------------------

    def set_target_range(self, min_t: float, max_t: float) -> None:
        """Update the target duration range (called by CurriculumCallback)."""
        self._min_target = float(min_t)
        self._max_target = float(max_t)

    # ------------------------------------------------------------------
    # City management
    # ------------------------------------------------------------------

    def _set_active_city(self, place: str) -> None:
        self.current_city_name = place
        data = self.city_data[place]

        self.G = data["G"]
        self.home_node = data["home_node"]
        self.base_pois: list = data["pois"]
        self.paths: dict = data["paths"]
        self.path_lengths: dict = data["path_lengths"]

        # Distance matrix (O(1) lookup)
        key_nodes = [self.home_node] + list(self.base_pois)
        n = len(key_nodes)
        self._kn_to_idx: dict = {node: i for i, node in enumerate(key_nodes)}
        self._home_idx: int = 0

        self._dist_matrix = np.full((n, n), np.inf, dtype=np.float32)
        for i, u in enumerate(key_nodes):
            pl_u = self.path_lengths.get(u, {})
            for j, v in enumerate(key_nodes):
                if v in pl_u:
                    self._dist_matrix[i, j] = float(pl_u[v])
        self._dist_matrix[np.arange(n), np.arange(n)] = 0.0

        self._home_coords = np.array(
            [self.G.nodes[self.home_node]["x"], self.G.nodes[self.home_node]["y"]],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Prioritised city sampling
        weights = np.array(
            [self.city_error_ewma[p] for p in self.places], dtype=np.float64
        )
        weights /= weights.sum()
        city = np.random.choice(self.places, p=weights)
        self._set_active_city(city)

        # Target duration (curriculum-controlled range)
        if options and "target_duration" in options:
            self.target_duration = float(options["target_duration"])
        else:
            self.target_duration = float(
                np.random.uniform(self._min_target, self._max_target)
            )

        # Episode POIs (shuffled each episode for generalisation)
        self.episode_pois: list = list(self.base_pois)
        np.random.shuffle(self.episode_pois)

        self._ep_poi_indices = np.array(
            [self._kn_to_idx[p] for p in self.episode_pois], dtype=np.int32
        )
        self._ep_poi_coords = np.array(
            [[self.G.nodes[p]["x"], self.G.nodes[p]["y"]] for p in self.episode_pois],
            dtype=np.float32,
        )
        self._ep_poi_home_dist = self._dist_matrix[
            self._ep_poi_indices, self._home_idx
        ]

        # Episode state
        self.current_duration: float = 0.0
        self.current_node = self.home_node
        self._current_idx: int = self._home_idx

        self.visited_nodes: set = {self.home_node}
        self.visited_edges: set = set()
        self._visited_mask_np = np.zeros(self.num_pois, dtype=np.float32)
        self.visited_pois_mask: list[int] = [0] * self.num_pois
        self.episode_path: list = [self.home_node]
        self.step_count: int = 0
        self.breadcrumbs: list = [self.home_node, self.home_node, self.home_node]

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        curr_x = float(self.G.nodes[self.current_node]["x"])
        curr_y = float(self.G.nodes[self.current_node]["y"])
        curr_xy = np.array([curr_x, curr_y], dtype=np.float32)

        target_norm = self.target_duration / self.MAX_TARGET_DURATION
        current_norm = self.current_duration / self.MAX_TARGET_DURATION

        dist_to_home = float(self._dist_matrix[self._current_idx, self._home_idx])
        time_to_home = dist_to_home / self.WALKING_SPEED_M_MIN
        time_to_home_norm = time_to_home / self.MAX_TARGET_DURATION

        delta_real = self.target_duration - self.current_duration
        delta_norm = float(np.clip(delta_real / self.MAX_TARGET_DURATION, -1.0, 1.0))
        urgency_norm = (
            1.0 if delta_real <= 0.0
            else float(min(1.0, time_to_home / (max(0.1, delta_real) * 10.0)))
        )

        scalars = np.array(
            [target_norm, current_norm, time_to_home_norm, delta_norm, urgency_norm],
            dtype=np.float32,
        )

        home_vec = (self._home_coords - curr_xy) / self.MAX_COORD_DIFF

        bc_xy = np.array(
            [[self.G.nodes[n]["x"], self.G.nodes[n]["y"]] for n in self.breadcrumbs],
            dtype=np.float32,
        )
        bc_rel = (bc_xy - curr_xy) / self.MAX_COORD_DIFF

        visited = self._visited_mask_np
        dx = (self._ep_poi_coords[:, 0] - curr_x) / self.MAX_COORD_DIFF
        dy = (self._ep_poi_coords[:, 1] - curr_y) / self.MAX_COORD_DIFF

        dist_to_pois = self._dist_matrix[self._current_idx, self._ep_poi_indices]
        t_to_pois_min = dist_to_pois / self.WALKING_SPEED_M_MIN
        time_to_poi_norm = t_to_pois_min / self.MAX_TARGET_DURATION

        # Round-trip feasibility feature (key for agent to assess reachability)
        t_back_min = self._ep_poi_home_dist / self.WALKING_SPEED_M_MIN
        feasibility = np.clip(
            1.0 - (t_to_pois_min + t_back_min) / max(delta_real, 1.0),
            0.0, 1.0,
        ).astype(np.float32)

        not_visited = 1.0 - visited
        dx = dx * not_visited
        dy = dy * not_visited
        time_to_poi_norm = time_to_poi_norm * not_visited
        feasibility = feasibility * not_visited

        poi_features = np.stack(
            [visited, dx, dy, time_to_poi_norm, feasibility], axis=1
        ).ravel()

        return np.concatenate(
            [scalars, home_vec, bc_rel.ravel(), poi_features]
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Action masks
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.num_pois + 1, dtype=np.bool_)

        if self.cfg.masking_mode == "disabled":
            return mask

        time_left = self.target_duration - self.current_duration
        t_home = (
            float(self._dist_matrix[self._current_idx, self._home_idx])
            / self.WALKING_SPEED_M_MIN
        )

        # Emergency return (ontological + full modes)
        if self.cfg.masking_mode != "no_emergency":
            if t_home >= (time_left - 2.0) and self.step_count >= 1:
                return np.array(
                    [False] * self.num_pois + [True], dtype=np.bool_
                )

        # Ontological: mask visited POIs and current-position duplicates
        mask[: self.num_pois] &= ~(self._visited_mask_np.astype(np.bool_))
        mask[: self.num_pois] &= self._ep_poi_indices != self._current_idx

        # Require at least one POI step before going home
        if self.step_count < 1:
            mask[self.num_pois] = False

        if self.cfg.masking_mode in ("ontological", "no_emergency"):
            if not mask[: self.num_pois].any():
                mask[self.num_pois] = True  # fallback: allow home when all POIs done
            return mask

        # Temporal look-ahead (only 'full' mode)
        t_to_pois = (
            self._dist_matrix[self._current_idx, self._ep_poi_indices]
            / self.WALKING_SPEED_M_MIN
        )
        t_back = self._ep_poi_home_dist / self.WALKING_SPEED_M_MIN
        too_far = (t_to_pois + t_back) > (time_left + 5.0)
        mask[: self.num_pois] &= ~too_far

        if not mask[: self.num_pois].any():
            mask[self.num_pois] = True
        return mask

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int):
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        is_home_action = action == self.num_pois

        # Penalty for revisiting in disabled-masking mode
        if self.cfg.masking_mode == "disabled" and not is_home_action:
            if self.visited_pois_mask[action] == 1:
                info = self._make_info()
                info["invalid_action"] = True
                return self._get_obs(), -10.0, False, False, info

        next_node = (
            self.home_node if is_home_action else self.episode_pois[action]
        )
        if not is_home_action:
            self.visited_pois_mask[action] = 1
            self._visited_mask_np[action] = 1.0

        next_idx = self._kn_to_idx[next_node]
        path = self.paths[self.current_node][next_node]
        dist = float(self._dist_matrix[self._current_idx, next_idx])

        if len(path) == 0 and self.current_node != next_node:
            info = self._make_info()
            info["error"] = "no_path"
            return self._get_obs(), -10.0, True, False, info

        time_taken = dist / self.WALKING_SPEED_M_MIN
        self.current_duration += time_taken

        # ---- Step shaping (v2: balanced exploration) ----
        c = self.cfg
        step_shaping = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = tuple(sorted((u, v)))
            if edge in self.visited_edges:
                step_shaping += c.w_duplicate_edge
            else:
                self.visited_edges.add(edge)

            if v not in self.visited_nodes:
                reward += c.w_new_node          # +0.3 (was +0.01)
                self.visited_nodes.add(v)
            else:
                reward += c.w_revisit_node      # -0.3 (was -1.0)

        step_shaping = max(c.step_shaping_cap, step_shaping)
        reward += step_shaping

        if path:
            self.episode_path.extend(path[1:])

        self.current_node = next_node
        self._current_idx = next_idx
        self.breadcrumbs.pop(0)
        self.breadcrumbs.append(next_node)

        # ---- Dense temporal shaping (potential-based, Ng et al. 1999) ----
        # Give +w per step when projected return time is near target.
        # This provides a continuous gradient for time management.
        if not is_home_action:
            t_home_now = (
                float(self._dist_matrix[self._current_idx, self._home_idx])
                / self.WALKING_SPEED_M_MIN
            )
            projected_end = self.current_duration + t_home_now
            proj_err = abs(self.target_duration - projected_end) / max(
                self.target_duration, 1e-6
            )
            # Linear decay: +w_temporal_dense at 0% proj error, 0 at ≥50% error
            dense = c.w_temporal_dense * max(0.0, 1.0 - 2.0 * proj_err)
            reward += dense

            # Safety penalty: agent stranded too far from home
            remaining = self.target_duration - self.current_duration
            if t_home_now > remaining + 5.0 and self.step_count >= 2:
                reward -= 0.5

        # ---- Terminal rewards ----
        reward_time = 0.0
        reward_circularity = 0.0

        if is_home_action:
            terminated = True
            error_ratio = abs(self.target_duration - self.current_duration) / max(
                self.target_duration, 1e-6
            )

            reward_time = max(
                c.w_terminal_time_cap,
                c.w_terminal_time_max - error_ratio * (c.w_terminal_time_max * 2.0),
            )
            reward += reward_time

            # Circularity bonus: always proportional (no hard gate)
            if c.use_circularity_bonus and len(self.episode_path) > 5:
                circ_score = self._compute_circularity_score()
                # timing_factor: 1.0 at 0% error, 0.0 at ≥50% error
                timing_factor = max(0.0, 1.0 - error_ratio / c.circularity_error_gate)
                reward_circularity = circ_score * c.w_circularity_max * timing_factor
                reward += reward_circularity

            # Update EWMA difficulty tracker
            self.city_error_ewma[self.current_city_name] = (
                0.95 * self.city_error_ewma[self.current_city_name] + 0.05 * error_ratio
            )

        # Truncation
        if self.step_count >= self.max_steps or self.current_duration > (
            self.target_duration * 2 + 20
        ):
            truncated = True
            if not is_home_action:
                reward -= 5.0

        info = self._make_info()
        info["reward_time"] = reward_time
        info["reward_circularity"] = reward_circularity
        info["reward_step_shaping"] = step_shaping

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_info(self) -> dict:
        return {
            "current_city": self.current_city_name,
            "current_duration": self.current_duration,
            "target_duration": self.target_duration,
            "episode_path": list(self.episode_path),
            "masking_mode": self.cfg.masking_mode,
        }

    def _compute_circularity_score(self) -> float:
        """Shoelace-based circularity in [0, 1]."""
        xs = np.array(
            [self.G.nodes[n]["x"] for n in self.episode_path], dtype=np.float64
        )
        ys = np.array(
            [self.G.nodes[n]["y"] for n in self.episode_path], dtype=np.float64
        )
        mean_lat = float(np.mean(ys))
        xs = xs * np.cos(np.radians(mean_lat))

        area_deg2 = 0.5 * abs(
            float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
        )
        area_sqm = area_deg2 * (111_111.0 ** 2)
        perimeter = self.current_duration * self.WALKING_SPEED_M_MIN
        ideal_area = (perimeter ** 2) / (4 * np.pi)
        return float(min(1.0, area_sqm / max(1.0, ideal_area)))
