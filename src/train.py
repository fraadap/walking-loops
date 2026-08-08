"""
Training entry-point for WalkingEnvV2 with attention policy.

Key differences from v1:
  - Curriculum learning: target range anneals from [60,120] → [30,120] → [10,120]
  - Attention policy: WalkingAttentionPolicy (Pointer Network style)
  - n_envs=16 (RTX 3090, Linux) — reduce to 2 for laptop testing
  - learning_rate=1e-4 (attention models benefit from smaller lr)
  - batch_size=512 (larger batches stabilise attention gradient estimates)
  - 5M total timesteps (more room for curriculum phases)

GPU note:
  Observations are already float32 numpy arrays from the env.
  The policy runs on torch.device('cuda') when available.
  SubprocVecEnv uses fork() on Linux → fast multi-env collection.

Usage:
    python -m walking_loops.train                           # full 5M run
    python -m walking_loops.train --debug                   # 200k laptop debug run
    python -m walking_loops.train --timesteps 1000000       # custom budget
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import Config, EnvConfig, TrainConfig
from env import WalkingEnvV2
from policy import WalkingAttentionPolicy, PolicyConfig


# ---------------------------------------------------------------------------
# RTX 3090 / Ampere GPU setup
# ---------------------------------------------------------------------------

def _setup_cuda() -> None:
    """
    Apply RTX 3090 (Ampere SM 86) specific CUDA optimisations.
    Safe no-op when no GPU is present.

    TF32:
      Ampere tensor cores can compute float32 matmuls using TF32 accumulation
      (10-bit mantissa vs 23-bit), giving ~3x throughput with negligible (<0.01%)
      accuracy difference for RL training.
      Enabled via allow_tf32 flags AND set_float32_matmul_precision('high').

    cuDNN benchmark:
      Runs a short autotuning pass on the first forward call to find the fastest
      convolution / attention algorithm for the fixed input shapes.
      Essential when input shapes are constant (our case: fixed N=80, d=128).
    """
    if not torch.cuda.is_available():
        print("[CUDA] No GPU found — running on CPU")
        return

    # TF32 for matrix multiplications (PyTorch >= 1.9)
    torch.set_float32_matmul_precision("high")          # enables TF32 globally
    torch.backends.cuda.matmul.allow_tf32 = True        # cuBLAS matmuls
    torch.backends.cudnn.allow_tf32 = True               # cuDNN convolutions

    # cuDNN autotuner — finds fastest algo for the fixed attention shapes
    torch.backends.cudnn.benchmark = True

    dev = torch.cuda.get_device_properties(0)
    vram_gb = dev.total_memory / 1e9
    print(f"[CUDA] Device : {dev.name}")
    print(f"[CUDA] VRAM   : {vram_gb:.1f} GB")
    print(f"[CUDA] SM     : {dev.major}.{dev.minor}")
    print("[CUDA] TF32 enabled — ~3x matmul throughput on Ampere")
    print("[CUDA] cuDNN benchmark enabled — autotuning attention kernels")


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(
    places: list[str],
    cfg: EnvConfig,
) -> WalkingEnvV2:
    return Monitor(WalkingEnvV2(places=places, cfg=cfg))


def _preload_cities(places: list[str], cfg: EnvConfig) -> None:
    """
    Load/download all city graphs in the MAIN process before spawning
    SubprocVecEnv — same pipeline as v1's single WalkingEnvV2 instance.

    Creates a temporary env with all cities, which triggers get_or_create_graph
    for each (download if not cached, load from cache otherwise), then discards it.
    Child processes created by SubprocVecEnv will subsequently find every graph
    already on disk and skip any network call.
    """
    all_cities = list(dict.fromkeys(places))
    print(f"[train] Pre-loading {len(all_cities)} city graph(s) in main process …")
    tmp_env = WalkingEnvV2(places=all_cities, cfg=cfg)
    tmp_env.close()
    print("[train] All city graphs ready.\n")


def make_vec_env(places: list[str], cfg: EnvConfig, n_envs: int) -> SubprocVecEnv:
    """Create SubprocVecEnv (no VecNormalize — obs already normalised)."""
    env_fn = partial(_make_env, places=places, cfg=cfg)
    return SubprocVecEnv([env_fn for _ in range(n_envs)])


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """
    Adjusts the target duration range in all parallel envs according to a
    three-phase curriculum:

      Phase 1 (0 → phase2_start):       targets in [60, 120] min  (easy: long loops)
      Phase 2 (phase2_start → phase3):  targets in [30, 120] min  (medium)
      Phase 3 (phase3 → end):           targets in [10, 120] min  (full range)

    Each env exposes `set_target_range(min_t, max_t)` (WalkingEnvV2 API).
    """

    def __init__(self, cfg: Config, verbose: int = 0):
        super().__init__(verbose)
        self._cfg = cfg
        self._current_phase: int = 1

    def _update_range(self, min_t: float, max_t: float) -> None:
        if self.verbose > 0:
            print(
                f"[Curriculum] step={self.num_timesteps:,} → "
                f"target range [{min_t:.0f}, {max_t:.0f}] min"
            )
        # SubprocVecEnv has no `.envs` attribute; dispatch into each worker
        # process via env_method (forwarded through the Monitor wrapper).
        self.training_env.env_method("set_target_range", min_t, max_t)

    def _on_step(self) -> bool:
        t = self.num_timesteps
        tc = self._cfg.train

        if self._current_phase == 1 and t >= tc.curriculum_phase2_start:
            self._current_phase = 2
            self._update_range(30.0, 120.0)
        elif self._current_phase == 2 and t >= tc.curriculum_phase3_start:
            self._current_phase = 3
            self._update_range(
                self._cfg.env.final_min_target, self._cfg.env.final_max_target
            )
        return True


# ---------------------------------------------------------------------------
# Reward logger callback
# ---------------------------------------------------------------------------

class RewardLoggerCallback(BaseCallback):
    """Collects per-episode reward decomposition for learning curve plots."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.records: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            td = info.get("target_duration", 1e-6)
            ad = info.get("current_duration", 0.0)
            self.records.append(
                {
                    "timestep": self.num_timesteps,
                    "reward_total": float(info["episode"]["r"]),
                    "reward_time": float(info.get("reward_time", 0.0)),
                    "reward_circularity": float(info.get("reward_circularity", 0.0)),
                    "reward_step_shaping": float(info.get("reward_step_shaping", 0.0)),
                    "ep_len": int(info["episode"]["l"]),
                    "target_duration": float(td),
                    "actual_duration": float(ad),
                    "pct_error": abs(td - ad) / max(td, 1e-6),
                    "city": info.get("current_city", ""),
                }
            )
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Config | None = None) -> float:
    """
    Train the attention policy.

    Returns:
        mean_pct_error on final evaluation set (lower = better).
    """
    cfg = cfg or Config.default()
    tc = cfg.train

    _setup_cuda()  # TF32 + cuDNN benchmark (must run before model creation)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    print(f"[train] Config: {tc.total_timesteps:,} steps, n_envs={tc.n_envs}")

    # ---- Pre-load all graphs in the MAIN process (v1 pipeline) ----
    # A single temporary env loads/downloads every city before SubprocVecEnv
    # spawns child processes. Children then find all graphs cached on disk.
    all_places = list(
        dict.fromkeys(list(tc.training_cities) + list(tc.eval_cities))
    )
    _preload_cities(all_places, cfg.env)

    # ---- Envs ----
    vec_env = make_vec_env(
        places=list(tc.training_cities), cfg=cfg.env, n_envs=tc.n_envs
    )
    eval_env = WalkingEnvV2(places=list(tc.eval_cities), cfg=cfg.env)

    # Start curriculum at phase 1 (long targets only).
    # SubprocVecEnv runs each env in a worker process and exposes no `.envs`
    # list — methods must be dispatched with env_method(). The call is forwarded
    # through the Monitor wrapper to WalkingEnvV2.set_target_range in each worker.
    vec_env.env_method(
        "set_target_range",
        cfg.env.initial_min_target,
        cfg.env.initial_max_target,
    )

    # ---- Policy ----
    _is_cuda = torch.cuda.is_available()
    policy_kwargs = dict(
        num_pois=cfg.env.num_pois,
        policy_cfg=cfg.policy,
        use_amp=_is_cuda,           # BF16 autocast — free ~2x on Ampere, off on CPU
        compile_encoder=_is_cuda,   # torch.compile — slow first call, ~30% gain after
    )

    model = MaskablePPO(
        WalkingAttentionPolicy,
        vec_env,
        verbose=1,
        learning_rate=tc.learning_rate,
        n_steps=tc.n_steps,
        batch_size=tc.batch_size,
        n_epochs=tc.n_epochs,
        gamma=tc.gamma,
        gae_lambda=tc.gae_lambda,
        ent_coef=tc.ent_coef,
        clip_range=tc.clip_range,
        vf_coef=tc.vf_coef,
        policy_kwargs=policy_kwargs,
        device=device,
        tensorboard_log=tc.tensorboard_log,
    )

    print(f"[train] Policy parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # ---- Callbacks ----
    reward_logger = RewardLoggerCallback()
    curriculum = CurriculumCallback(cfg, verbose=1)
    callbacks = [reward_logger, curriculum]

    # ---- Training ----
    t0 = time.time()
    model.learn(
        total_timesteps=tc.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"[train] Training completed in {elapsed / 3600:.2f} h")

    # ---- Persist ----
    os.makedirs(os.path.dirname(tc.model_save_path) or ".", exist_ok=True)
    model.save(tc.model_save_path)
    print(f"[train] Model saved → {tc.model_save_path}.zip")

    # ---- Final evaluation ----
    mean_error = _quick_eval(model, eval_env, n_episodes=50)
    print(f"[train] Final eval mean_pct_error = {mean_error * 100:.2f}%")

    vec_env.close()
    eval_env.close()

    # ---- Learning curves ----
    if reward_logger.records:
        _save_learning_curves(reward_logger.records, cfg)

    return mean_error


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _quick_eval(model: MaskablePPO, env: WalkingEnvV2, n_episodes: int = 50) -> float:
    errors = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            masks = env.action_masks()
            action, _ = model.predict(
                obs[np.newaxis], action_masks=masks[np.newaxis], deterministic=True
            )
            obs, _, terminated, truncated, info = env.step(int(action[0]))
            done = terminated or truncated
        td = info.get("target_duration", 1e-6)
        ad = info.get("current_duration", 0.0)
        errors.append(abs(td - ad) / max(td, 1e-6))
    return float(np.mean(errors)) if errors else 1.0


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def _save_learning_curves(records: list[dict], cfg: Config) -> None:
    os.makedirs("assets", exist_ok=True)
    df = pd.DataFrame(records)
    window = max(50, len(df) // 100)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for col, label, color in [
        ("reward_total", "Total", "C0"),
        ("reward_time", "Time precision", "C1"),
        ("reward_circularity", "Circularity", "C2"),
    ]:
        if col in df.columns:
            ema = df[col].rolling(window, min_periods=1).mean()
            ax.plot(df["timestep"], ema, label=label, color=color, linewidth=1.5)
    ax.set_title("Reward decomposition (EMA)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    df["success"] = (df["pct_error"] <= 0.10).astype(float)
    sr = df["success"].rolling(window, min_periods=1).mean()
    ax.plot(df["timestep"], sr, color="C3", linewidth=1.5)
    ax.set_title("Success rate (<= 10% error, rolling)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    bucket_order = ["10-30", "30-60", "60-90", "90-120"]
    df["bucket"] = df["target_duration"].apply(
        lambda t: "10-30" if t <= 30 else (
            "30-60" if t <= 60 else ("60-90" if t <= 90 else "90-120")
        )
    )
    grouped = df.groupby("bucket")["pct_error"].mean() * 100
    grouped = grouped.reindex(bucket_order, fill_value=float("nan"))
    grouped.plot.bar(ax=ax, color="C0", edgecolor="black", alpha=0.8)
    ax.axhline(10, color="r", linestyle="--", label="10% tol.")
    ax.set_title("Mean error by target bucket")
    ax.set_ylabel("% error")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if "city" in df.columns:
        city_err = df.groupby("city")["pct_error"].mean() * 100
        city_err.plot.bar(ax=ax, color="C4", edgecolor="black", alpha=0.8)
        ax.axhline(10, color="r", linestyle="--", label="10% tol.")
        ax.set_title("Mean error per training city")
        ax.set_ylabel("% error")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.suptitle(
        f"WalkingEnvV2 | Attention Policy | {cfg.train.total_timesteps // 1_000}k steps",
        y=1.01,
    )
    plt.tight_layout()
    path = "assets/learning_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train] Learning curve saved -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train WalkingEnvV2 attention agent")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--debug", action="store_true", help="Use laptop_debug config")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = Config.laptop_debug() if args.debug else Config.default()

    # CLI overrides
    if args.timesteps is not None:
        cfg = Config(
            env=cfg.env,
            policy=cfg.policy,
            train=TrainConfig(
                **{**cfg.train.__dict__, "total_timesteps": args.timesteps}
            ),
        )
    if args.n_envs is not None:
        cfg = Config(
            env=cfg.env,
            policy=cfg.policy,
            train=TrainConfig(
                **{**cfg.train.__dict__, "n_envs": args.n_envs}
            ),
        )

    mean_err = train(cfg)
    print(f"\nFinal mean_pct_error: {mean_err * 100:.2f}%")
