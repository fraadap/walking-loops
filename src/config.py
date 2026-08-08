"""
Centralised configuration for the walking-loops v2 project.

All hyperparameters live here as frozen dataclasses.  Pass a Config instance
to env, policy, and training functions instead of scattered keyword arguments.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvConfig:
    # Graph / geography
    num_pois: int = 80
    radius_meters: int = 3000
    masking_mode: str = "ontological"   # 'ontological' | 'full' | 'no_emergency' | 'disabled'
    use_circularity_bonus: bool = True

    # Speed
    walking_speed_kmh: float = 5.0

    # Episode
    max_steps: int = 60                 # increased from 50 → room for longer episodes
    max_target_duration: float = 120.0  # minutes
    max_coord_diff: float = 0.03        # ~3.3 km at 50°N

    # Reward weights (v2 — balanced 1:1 exploration ratio)
    w_new_node: float = 0.30            # per new node discovered (was 0.01 in v1)
    w_revisit_node: float = -0.30       # per revisited node (was -1.0 in v1)
    w_duplicate_edge: float = -0.10     # per duplicate edge (was -0.5 in v1)
    step_shaping_cap: float = -3.0      # minimum step shaping reward (was -10 in v1)
    w_temporal_dense: float = 0.10      # per-step reward for good projected timing
    w_terminal_time_max: float = 100.0  # max terminal time reward
    w_terminal_time_cap: float = -30.0  # min terminal time reward (clamp)
    w_circularity_max: float = 25.0     # max circularity bonus
    circularity_error_gate: float = 0.50  # error above which circularity bonus = 0

    # Curriculum (set by callback at runtime via env.set_target_range)
    initial_min_target: float = 60.0    # curriculum start: only long targets
    initial_max_target: float = 120.0
    final_min_target: float = 10.0      # curriculum end: full range
    final_max_target: float = 120.0


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyConfig:
    # Attention encoder
    d_model: int = 128              # embedding dimension
    n_heads: int = 8                # number of attention heads
    dropout: float = 0.0            # attention dropout (set >0 for regularisation)
    logit_clip_C: float = 10.0      # tanh clipping constant (Bello et al. 2016)

    # Critic MLP (on top of attended context)
    n_critic_layers: int = 2
    critic_hidden: int = 256

    # Obs layout (must match WalkingEnvV2)
    context_dim: int = 13           # 5 scalars + 2 home_vec + 6 breadcrumbs
    poi_feature_dim: int = 5        # visited, dx, dy, t_norm, feasibility


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    # Algorithm
    learning_rate: float = 1e-4        # lower than default; attention models prefer small lr
    n_steps: int = 2048                # rollout steps per env per iteration
    batch_size: int = 512              # large batch for stable attention gradients
    n_epochs: int = 5                  # fewer epochs prevent overfitting with attention
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01             # low — attention explores via attention diversity
    clip_range: float = 0.2
    vf_coef: float = 0.5

    # Environment parallelism
    n_envs: int = 16                   # for RTX 3090 Linux training; reduce for laptop testing

    # Training duration
    total_timesteps: int = 5_000_000

    # Curriculum schedule (absolute timesteps)
    curriculum_phase2_start: int = 500_000    # switch to [30, 120]
    curriculum_phase3_start: int = 1_500_000  # switch to [10, 120]

    # Persistence
    model_save_path: str = "models/ppo_walking_v2"
    tensorboard_log: str = "./tensorboard_logs/"

    # Cities
    # Cities seen during training. Adding more improves generalisation but each
    # one needs its OSMnx graph downloaded once into cache/ (internet required).
    training_cities: tuple[str, ...] = (
        "Brussels, Belgium",
        "Barcelona, Spain",
        "Vienna, Austria",
        "Lisbon, Portugal",
        "Amsterdam, Netherlands",
    )
    # Held-out city used for the quick in-training sanity check.
    # The full zero-shot benchmark lives in eval.py / metrics.EVAL_CITIES.
    eval_cities: tuple[str, ...] = (
        "Florence, Italy",
    )


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def default(cls) -> "Config":
        return cls()

    @classmethod
    def laptop_debug(cls) -> "Config":
        """Reduced config for fast iteration on CPU laptop."""
        return cls(
            env=EnvConfig(num_pois=20, radius_meters=2000),
            policy=PolicyConfig(d_model=64, n_heads=4, n_critic_layers=1, critic_hidden=128),
            train=TrainConfig(
                n_envs=2,
                total_timesteps=200_000,
                batch_size=128,
                curriculum_phase2_start=50_000,
                curriculum_phase3_start=100_000,
            ),
        )
