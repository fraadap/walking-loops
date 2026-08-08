"""
Attention-based actor-critic policy for WalkingEnvV2.

Architecture: Pointer Network (Vinyals et al. 2015) combined with cross-attention
(Kool et al. 2019 "Attention, Learn to Solve Routing Problems").

Key properties:
  - Permutation-invariant: POI ordering at reset doesn't affect policy output
  - Each POI is encoded independently with a shared MLP
  - Context (current state) attends over POI embeddings via multi-head attention
  - Compatibility scores (logits) = C * tanh(W_Q h_ctx · W_K h_poi / sqrt(d))
  - Integration with SB3/MaskablePPO via _build_mlp_extractor override

SB3 integration strategy (minimal override):
  1. _build_mlp_extractor: creates AttentionMlpExtractor (replaces default MLP)
  2. _get_action_dist_from_latent: bypasses action_net (logits come directly from attention)
  3. All other methods (forward, evaluate_actions, get_distribution, predict_values)
     are inherited and work correctly with the above two overrides.

GPU optimizations (RTX 3090 / Ampere):
  - use_amp=True:  BFloat16 autocast in evaluate_actions / get_distribution / predict_values.
                   BF16 (not FP16) is used — Ampere supports it natively and it requires
                   NO GradScaler (same dynamic range as FP32, just lower mantissa precision).
                   Yields ~2x speedup on attention matmuls with negligible accuracy loss.
  - compile_encoder=True:  torch.compile(mode='reduce-overhead') on the AttentionEncoder.
                   Fuses elementwise ops, eliminates Python overhead between kernel launches.
                   First call is slow (~30s compilation); subsequent calls are 20-40% faster.
"""
from __future__ import annotations

import contextlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import MaskableDistribution

from config import PolicyConfig


def _autocast_ctx(enabled: bool, device_is_cuda: bool):
    """BF16 autocast context — no-op on CPU or when disabled."""
    if enabled and device_is_cuda:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Attention encoder
# ---------------------------------------------------------------------------

class AttentionEncoder(nn.Module):
    """
    Encodes the flat WalkingEnvV2 observation into:
      - action logits (N+1,)  — via Pointer Network compatibility scores
      - value features (d,)   — attended context for critic

    Observation layout expected:
      obs[:context_dim]                    — global context (13 dims)
      obs[context_dim:].view(B, N, 5)      — per-POI features (N×5 dims)

    Architecture:
      context → LayerNorm → Linear → ReLU → Linear  (context embedding)
      POI_i   → Linear → LayerNorm → ReLU  (shared POI encoder)
      Cross-attention: Q = context_emb, K = V = poi_embs
      Compatibility: C * tanh( W_Q h_ctx · W_K h_poi_i / sqrt(d) )
      home logit: Linear(attended_ctx)
    """

    def __init__(self, cfg: PolicyConfig, num_pois: int):
        super().__init__()
        self.cfg = cfg
        self.num_pois = num_pois
        d = cfg.d_model

        # Context encoder (global state → d-dim embedding)
        self.context_encoder = nn.Sequential(
            nn.Linear(cfg.context_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

        # Shared POI encoder (applied independently to each POI's features)
        self.poi_encoder = nn.Sequential(
            nn.Linear(cfg.poi_feature_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Multi-head cross-attention: context queries POI embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )

        # Pointer network projections for compatibility scores
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)

        # Separate logit head for home action
        self.home_head = nn.Linear(d, 1)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, 13 + N*5) flat observation

        Returns:
            attended:   (B, d) — for critic and home logit
            poi_logits: (B, N) — compatibility score per POI
            home_logit: (B, 1) — logit for home action
        """
        B = obs.shape[0]
        d = self.cfg.d_model
        N = self.num_pois

        # Split observation
        context = obs[:, : self.cfg.context_dim]           # (B, 13)
        poi_flat = obs[:, self.cfg.context_dim:]           # (B, N*5)
        poi_feats = poi_flat.view(B, N, self.cfg.poi_feature_dim)  # (B, N, 5)

        # Encode
        ctx_emb = self.context_encoder(context)            # (B, d)
        poi_embs = self.poi_encoder(poi_feats)             # (B, N, d)

        # Cross-attention: context (as single query) attends over POIs
        q = ctx_emb.unsqueeze(1)                          # (B, 1, d)
        attended_out, _ = self.cross_attn(q, poi_embs, poi_embs)  # (B, 1, d)
        attended = attended_out.squeeze(1) + ctx_emb      # (B, d) — residual

        # Pointer network compatibility scores with logit clipping (Bello et al. 2016)
        q_proj = self.W_Q(attended).unsqueeze(1)          # (B, 1, d)
        k_proj = self.W_K(poi_embs)                       # (B, N, d)
        scale = d ** 0.5
        raw_scores = torch.bmm(q_proj, k_proj.transpose(1, 2)).squeeze(1) / scale  # (B, N)
        poi_logits = self.cfg.logit_clip_C * torch.tanh(raw_scores)  # (B, N)

        # Home action logit (separate head)
        home_logit = self.home_head(attended)             # (B, 1)

        return attended, poi_logits, home_logit


# ---------------------------------------------------------------------------
# SB3-compatible MlpExtractor wrapper
# ---------------------------------------------------------------------------

class AttentionMlpExtractor(nn.Module):
    """
    Replaces the default SB3 MlpExtractor.

    Interface contract (SB3 2.x):
      - self.latent_dim_pi: int   — size of actor latent (= N+1, the action logits)
      - self.latent_dim_vf: int   — size of critic latent (= d_model)
      - forward(features) -> (latent_pi, latent_vf)
      - forward_actor(features) -> latent_pi
      - forward_critic(features) -> latent_vf
    """

    def __init__(self, cfg: PolicyConfig, num_pois: int):
        super().__init__()
        self.attention = AttentionEncoder(cfg, num_pois)
        self.latent_dim_pi: int = num_pois + 1    # action logits (N POIs + home)
        self.latent_dim_vf: int = cfg.d_model     # attended context for critic

    def _encode(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits (B, N+1), value_features (B, d))."""
        attended, poi_logits, home_logit = self.attention(features)
        action_logits = torch.cat([poi_logits, home_logit], dim=1)
        return action_logits, attended

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        logits, _ = self._encode(features)
        return logits

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        _, value_feats = self._encode(features)
        return value_feats


# ---------------------------------------------------------------------------
# Custom actor-critic policy
# ---------------------------------------------------------------------------

class WalkingAttentionPolicy(MaskableActorCriticPolicy):
    """
    MaskableActorCriticPolicy with Pointer-Network-based actor.

    GPU optimizations (enabled by default for RTX 3090):
      use_amp=True       — BFloat16 autocast in all forward passes (no GradScaler needed)
      compile_encoder=True — torch.compile(mode='reduce-overhead') on AttentionEncoder

    Usage:
        model = MaskablePPO(
            WalkingAttentionPolicy,
            env,
            policy_kwargs={
                "num_pois": 80,
                "policy_cfg": PolicyConfig(),
                "use_amp": True,
                "compile_encoder": True,
            },
        )
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        num_pois: int = 80,
        policy_cfg: Optional[PolicyConfig] = None,
        use_amp: bool = True,
        compile_encoder: bool = True,
        **kwargs,
    ):
        self._num_pois = num_pois
        self._policy_cfg = policy_cfg or PolicyConfig()
        self._use_amp = use_amp
        self._compile_encoder = compile_encoder
        # Disable SB3's default MLP extractor.
        kwargs.setdefault("net_arch", [])
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # ------------------------------------------------------------------
    # Override 1: replace MLP extractor with attention network
    # ------------------------------------------------------------------
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AttentionMlpExtractor(
            cfg=self._policy_cfg,
            num_pois=self._num_pois,
        )
        # torch.compile: fuses ops, eliminates Python overhead between kernels.
        # Compiles only the inner AttentionEncoder so the extractor wrapper
        # remains normally picklable (SB3 model.save() uses pickle).
        if self._compile_encoder and torch.cuda.is_available() and hasattr(torch, "compile"):
            self.mlp_extractor.attention = torch.compile(
                self.mlp_extractor.attention,
                mode="reduce-overhead",
                fullgraph=False,
            )

    # ------------------------------------------------------------------
    # Override 2: bypass action_net (logits come directly from attention)
    # ------------------------------------------------------------------
    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor
    ) -> MaskableDistribution:
        return self.action_dist.proba_distribution(action_logits=latent_pi)

    # ------------------------------------------------------------------
    # Override 3-5: BF16 autocast wrappers for the three SB3 hotpaths
    # ------------------------------------------------------------------

    def evaluate_actions(self, obs, actions, action_masks=None):
        """PPO update path — called once per mini-batch during policy.train()."""
        is_cuda = obs.is_cuda if isinstance(obs, torch.Tensor) else False
        with _autocast_ctx(self._use_amp, is_cuda):
            values, log_prob, entropy = super().evaluate_actions(
                obs, actions, action_masks
            )
        # Cast back to FP32 outside autocast: the BF16 outputs would otherwise
        # propagate into the loss (mixing with FP32 advantages/returns) and into
        # numpy conversions that do not support BFloat16.
        values = values.float()
        log_prob = log_prob.float()
        if entropy is not None:
            entropy = entropy.float()
        return values, log_prob, entropy

    def get_distribution(self, obs, action_masks=None):
        """Rollout collection path — called to sample actions from each env step."""
        is_cuda = obs.is_cuda if isinstance(obs, torch.Tensor) else False
        with _autocast_ctx(self._use_amp, is_cuda):
            return super().get_distribution(obs, action_masks)

    def predict_values(self, obs):
        """Value estimation path — called for GAE advantage computation."""
        is_cuda = obs.is_cuda if isinstance(obs, torch.Tensor) else False
        with _autocast_ctx(self._use_amp, is_cuda):
            values = super().predict_values(obs)
        # SB3 stores these in the rollout buffer via .cpu().numpy(), which has no
        # BFloat16 dtype — cast back to FP32 after leaving the autocast region.
        return values.float()
