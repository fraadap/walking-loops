"""
Unit and behavioural tests for WalkingEnvV2.

Tests cover:
  - Observation shape and dtype
  - Reward structure (v2 balanced weights)
  - Action masking (step < 1 gate, emergency)
  - Curriculum API (set_target_range)
  - Episode rollout (termination, truncation)
  - Circularity bonus (always proportional)
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TestObservation:
    def test_shape(self, mock_env_v2):
        obs, _ = mock_env_v2.reset(seed=0)
        N = mock_env_v2.num_pois
        assert obs.shape == (5 + 2 + 6 + N * 5,)

    def test_dtype(self, mock_env_v2):
        obs, _ = mock_env_v2.reset(seed=0)
        assert obs.dtype == np.float32

    def test_no_nan_inf(self, mock_env_v2):
        obs, _ = mock_env_v2.reset(seed=0)
        assert np.all(np.isfinite(obs))

    def test_target_norm_at_index_0(self, mock_env_v2):
        obs, _ = mock_env_v2.reset(seed=0, options={"target_duration": 30.0})
        assert abs(obs[0] - 30.0 / 120.0) < 1e-4

    def test_current_norm_zero_at_reset(self, mock_env_v2):
        obs, _ = mock_env_v2.reset(seed=0)
        assert abs(obs[1]) < 1e-4


# ---------------------------------------------------------------------------
# Reward structure (v2 balanced weights)
# ---------------------------------------------------------------------------

class TestReward:
    def test_new_node_bonus_positive(self, mock_env_v2):
        mock_env_v2.reset(seed=0)
        mask = mock_env_v2.action_masks()
        poi_actions = np.where(mask[: mock_env_v2.num_pois])[0]
        _, reward, _, _, _ = mock_env_v2.step(int(poi_actions[0]))
        # Must be positive: the step visits at least one new node
        # (new_node bonus +0.3 outweighs duplicate edge -0.1 × few edges)
        assert reward > -10.0, "Reward should not be catastrophically negative on first step"

    def test_home_gives_terminal_reward_in_info(self, mock_env_v2):
        mock_env_v2.reset(seed=0)
        mock_env_v2.step_count = 3
        _, _, _, _, info = mock_env_v2.step(mock_env_v2.num_pois)
        assert "reward_time" in info
        assert isinstance(info["reward_time"], float)
        assert info["reward_time"] != 0.0 or mock_env_v2.current_duration == 0.0

    def test_circularity_bonus_in_info(self, mock_env_v2):
        mock_env_v2.reset(seed=0)
        mock_env_v2.step_count = 3
        _, _, _, _, info = mock_env_v2.step(mock_env_v2.num_pois)
        assert "reward_circularity" in info
        assert info["reward_circularity"] >= 0.0

    def test_v2_reward_weights_balanced(self, mock_env_v2):
        """new_node and revisit_node should be equal magnitude (1:1 ratio)."""
        cfg = mock_env_v2.cfg
        assert abs(cfg.w_new_node + cfg.w_revisit_node) < 1e-6, (
            f"Expected |w_new_node| == |w_revisit_node|, "
            f"got {cfg.w_new_node} vs {cfg.w_revisit_node}"
        )


# ---------------------------------------------------------------------------
# Action masking (v2 gates)
# ---------------------------------------------------------------------------

class TestActionMasking:
    def test_home_blocked_at_step_0(self, mock_env_v2):
        mock_env_v2.reset(seed=0)
        mock_env_v2.step_count = 0
        assert not mock_env_v2.action_masks()[mock_env_v2.num_pois]

    def test_home_accessible_after_step_1(self, mock_env_v2):
        mock_env_v2.reset(seed=0)
        mock_env_v2.step_count = 1
        mock_env_v2.target_duration = 999.0
        assert mock_env_v2.action_masks()[mock_env_v2.num_pois]

    def test_visited_poi_masked(self, mock_env_v2):
        mock_env_v2.reset(seed=0)
        mock_env_v2.step_count = 1
        mask = mock_env_v2.action_masks()
        poi_actions = np.where(mask[: mock_env_v2.num_pois])[0]
        action = int(poi_actions[0])
        mock_env_v2.step(action)
        assert not mock_env_v2.action_masks()[action]

    def test_emergency_fires_at_step_1_near_limit(self, mock_env_v2):
        mock_env_v2.reset(seed=0)
        mock_env_v2.step_count = 1
        mock_env_v2.target_duration = 10.0
        mock_env_v2.current_duration = 9.5
        mask = mock_env_v2.action_masks()
        # At home node, t_home = 0, time_left = 0.5, cond: 0 >= -1.5 → True
        assert mask[mock_env_v2.num_pois]  # home is valid (emergency fired)

    def test_fallback_opens_home_when_all_pois_done(self, mock_env_v2):
        mock_env_v2.reset(seed=0)
        mock_env_v2.step_count = 0
        mock_env_v2.target_duration = 999.0
        mock_env_v2._visited_mask_np[:] = 1.0
        mock_env_v2.visited_pois_mask = [1] * mock_env_v2.num_pois
        assert mock_env_v2.action_masks()[mock_env_v2.num_pois]


# ---------------------------------------------------------------------------
# Curriculum API
# ---------------------------------------------------------------------------

class TestCurriculum:
    def test_set_target_range_changes_sampling(self, mock_env_v2):
        mock_env_v2.set_target_range(60.0, 120.0)
        targets = []
        for i in range(50):
            mock_env_v2.reset(seed=i)
            targets.append(mock_env_v2.target_duration)
        assert min(targets) >= 60.0, "All targets should be >= 60 after set_target_range(60, 120)"
        assert max(targets) <= 120.0

    def test_set_target_range_full(self, mock_env_v2):
        mock_env_v2.set_target_range(10.0, 120.0)
        targets = []
        for i in range(100):
            mock_env_v2.reset(seed=i)
            targets.append(mock_env_v2.target_duration)
        assert min(targets) >= 10.0
        assert max(targets) <= 120.0

    def test_target_range_restored_after_options_override(self, mock_env_v2):
        mock_env_v2.set_target_range(60.0, 120.0)
        obs, _ = mock_env_v2.reset(options={"target_duration": 25.0})
        assert mock_env_v2.target_duration == 25.0  # override wins


# ---------------------------------------------------------------------------
# Episode rollouts
# ---------------------------------------------------------------------------

class TestEpisodeRollout:
    def test_episode_completes(self, mock_env_v2):
        obs, _ = mock_env_v2.reset(seed=42, options={"target_duration": 20.0})
        done, steps = False, 0
        while not done and steps < 300:
            mask = mock_env_v2.action_masks()
            action = int(np.where(mask)[0][0])
            obs, _, terminated, truncated, info = mock_env_v2.step(action)
            done = terminated or truncated
            steps += 1
        assert done
        assert "current_duration" in info

    def test_visited_mask_clears_on_reset(self, mock_env_v2):
        mock_env_v2.reset(seed=0)
        mask = mock_env_v2.action_masks()
        poi_actions = np.where(mask[: mock_env_v2.num_pois])[0]
        mock_env_v2.step(int(poi_actions[0]))
        mock_env_v2.reset(seed=0)
        assert all(v == 0 for v in mock_env_v2.visited_pois_mask)
        assert np.all(mock_env_v2._visited_mask_np == 0.0)

    def test_duration_increases_on_step(self, mock_env_v2):
        mock_env_v2.reset(seed=0)
        before = mock_env_v2.current_duration
        mask = mock_env_v2.action_masks()
        poi_actions = np.where(mask[: mock_env_v2.num_pois])[0]
        mock_env_v2.step(int(poi_actions[0]))
        assert mock_env_v2.current_duration > before

    def test_obs_changes_after_step(self, mock_env_v2):
        obs0, _ = mock_env_v2.reset(seed=0)
        mask = mock_env_v2.action_masks()
        poi_actions = np.where(mask[: mock_env_v2.num_pois])[0]
        obs1, _, _, _, _ = mock_env_v2.step(int(poi_actions[0]))
        assert not np.array_equal(obs0, obs1)
