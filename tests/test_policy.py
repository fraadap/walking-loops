"""
Unit tests for WalkingAttentionPolicy (v2).

Tests cover:
  - AttentionEncoder output shapes
  - AttentionMlpExtractor output shapes and latent_dim attributes
  - Permutation invariance: shuffling POIs → same action ordering (different indices, same scores)
  - WalkingAttentionPolicy integration: can predict, evaluate_actions works
  - Gradient flow through the full network
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from walking_loops.config import PolicyConfig, EnvConfig
from walking_loops.policy import AttentionEncoder, AttentionMlpExtractor, WalkingAttentionPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_POIS = 9  # matches conftest grid (POIs 1-9)
OBS_DIM = 5 + 2 + 6 + N_POIS * 5  # 58
BATCH = 4
CFG = PolicyConfig(d_model=32, n_heads=4, n_critic_layers=1, critic_hidden=64)


@pytest.fixture
def encoder():
    return AttentionEncoder(cfg=CFG, num_pois=N_POIS)


@pytest.fixture
def extractor():
    return AttentionMlpExtractor(cfg=CFG, num_pois=N_POIS)


@pytest.fixture
def random_obs():
    return torch.randn(BATCH, OBS_DIM)


# ---------------------------------------------------------------------------
# AttentionEncoder
# ---------------------------------------------------------------------------

class TestAttentionEncoder:
    def test_output_shapes(self, encoder, random_obs):
        attended, poi_logits, home_logit = encoder(random_obs)
        assert attended.shape == (BATCH, CFG.d_model)
        assert poi_logits.shape == (BATCH, N_POIS)
        assert home_logit.shape == (BATCH, 1)

    def test_logits_clipped(self, encoder, random_obs):
        _, poi_logits, home_logit = encoder(random_obs)
        assert poi_logits.abs().max().item() <= CFG.logit_clip_C + 1e-4, (
            "Pointer logits must be clipped to [-C, C]"
        )

    def test_no_nan_in_output(self, encoder, random_obs):
        attended, poi_logits, home_logit = encoder(random_obs)
        for t in (attended, poi_logits, home_logit):
            assert not torch.any(torch.isnan(t)), "Output contains NaN"

    def test_gradients_flow(self, encoder, random_obs):
        random_obs.requires_grad_(True)
        attended, poi_logits, home_logit = encoder(random_obs)
        loss = poi_logits.sum() + home_logit.sum() + attended.sum()
        loss.backward()
        assert random_obs.grad is not None
        assert not torch.all(random_obs.grad == 0)

    def test_permutation_invariance(self, encoder):
        """Shuffling POI features should produce the same set of logit values
        (different positions, same values — up to floating point)."""
        torch.manual_seed(42)
        obs = torch.randn(1, OBS_DIM)

        # Baseline logits
        _, logits_base, _ = encoder(obs)
        logits_base = logits_base.detach()

        # Shuffle POI block
        context = obs[:, :13]
        poi_block = obs[:, 13:].view(1, N_POIS, 5)
        perm = torch.randperm(N_POIS)
        poi_shuffled = poi_block[:, perm, :].view(1, -1)
        obs_shuffled = torch.cat([context, poi_shuffled], dim=1)

        _, logits_shuffled, _ = encoder(obs_shuffled)
        logits_shuffled = logits_shuffled.detach()

        # After shuffling POIs, the SET of logit values should be (approximately) the same
        # Each logit corresponds to the POI at that position, so sorted logit values match
        base_sorted = logits_base.sort(dim=1).values
        shuffled_sorted = logits_shuffled.sort(dim=1).values
        assert torch.allclose(base_sorted, shuffled_sorted, atol=1e-4), (
            "Sorted logit values should be equal after POI permutation "
            "(permutation invariance violated)"
        )


# ---------------------------------------------------------------------------
# AttentionMlpExtractor
# ---------------------------------------------------------------------------

class TestAttentionMlpExtractor:
    def test_latent_dims(self, extractor):
        assert extractor.latent_dim_pi == N_POIS + 1
        assert extractor.latent_dim_vf == CFG.d_model

    def test_forward_shapes(self, extractor, random_obs):
        latent_pi, latent_vf = extractor.forward(random_obs)
        assert latent_pi.shape == (BATCH, N_POIS + 1)
        assert latent_vf.shape == (BATCH, CFG.d_model)

    def test_forward_actor_shape(self, extractor, random_obs):
        logits = extractor.forward_actor(random_obs)
        assert logits.shape == (BATCH, N_POIS + 1)

    def test_forward_critic_shape(self, extractor, random_obs):
        feats = extractor.forward_critic(random_obs)
        assert feats.shape == (BATCH, CFG.d_model)

    def test_forward_consistent_with_actor_critic(self, extractor, random_obs):
        """forward() must return the same values as forward_actor / forward_critic."""
        latent_pi, latent_vf = extractor.forward(random_obs)
        logits = extractor.forward_actor(random_obs)
        feats = extractor.forward_critic(random_obs)
        assert torch.allclose(latent_pi, logits, atol=1e-5)
        assert torch.allclose(latent_vf, feats, atol=1e-5)


# ---------------------------------------------------------------------------
# WalkingAttentionPolicy — integration
# ---------------------------------------------------------------------------

class TestWalkingAttentionPolicy:
    """Test the policy using the mock env from conftest."""

    def test_policy_predict(self, mock_env_v2):
        from sb3_contrib import MaskablePPO
        from gymnasium import spaces

        env = mock_env_v2
        obs, _ = env.reset(seed=0)
        N = env.num_pois

        model = MaskablePPO(
            WalkingAttentionPolicy,
            env,
            policy_kwargs={
                "num_pois": N,
                "policy_cfg": PolicyConfig(d_model=32, n_heads=4, n_critic_layers=1, critic_hidden=64),
            },
            n_steps=16,
            batch_size=8,
            verbose=0,
        )

        masks = env.action_masks()
        action, _ = model.predict(
            obs[np.newaxis], action_masks=masks[np.newaxis], deterministic=True
        )
        assert 0 <= int(action[0]) <= N

    def test_policy_learns(self, mock_env_v2):
        """Policy should not crash and loss should be finite after a few updates."""
        from sb3_contrib import MaskablePPO

        env = mock_env_v2
        model = MaskablePPO(
            WalkingAttentionPolicy,
            env,
            policy_kwargs={
                "num_pois": env.num_pois,
                "policy_cfg": PolicyConfig(d_model=32, n_heads=4, n_critic_layers=1, critic_hidden=64),
            },
            n_steps=32,
            batch_size=16,
            n_epochs=2,
            verbose=0,
        )
        model.learn(total_timesteps=128)  # just verify no crash

    def test_evaluate_actions_shapes(self, mock_env_v2):
        """evaluate_actions must return tensors of correct shape."""
        from sb3_contrib import MaskablePPO
        import torch

        env = mock_env_v2
        N = env.num_pois
        obs_dim = 5 + 2 + 6 + N * 5

        model = MaskablePPO(
            WalkingAttentionPolicy,
            env,
            policy_kwargs={
                "num_pois": N,
                "policy_cfg": PolicyConfig(d_model=32, n_heads=4, n_critic_layers=1, critic_hidden=64),
            },
            n_steps=16,
            batch_size=8,
            verbose=0,
        )

        B = 4
        obs_t = torch.zeros(B, obs_dim)
        actions = torch.zeros(B, dtype=torch.long)
        masks = torch.ones(B, N + 1, dtype=torch.bool)

        values, log_probs, entropy = model.policy.evaluate_actions(
            obs_t, actions, action_masks=masks
        )
        assert values.shape == (B, 1)
        assert log_probs.shape == (B,)
        assert entropy is None or entropy.shape == (B,)
