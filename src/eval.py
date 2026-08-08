"""
Evaluation script for WalkingEnvV2.

Runs a deterministic test set (fixed seed=42) across multiple cities and
produces:
  - Per-city × per-bucket aggregates (CSV)
  - Hero plot (4 subplots)
  - Console report + statistics.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from metrics import compute_loop_metrics, make_deterministic_test_set, EVAL_CITIES
from config import Config, EnvConfig
from env import WalkingEnvV2
from policy import WalkingAttentionPolicy, PolicyConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_CITIES = EVAL_CITIES + [
    "Bruges, Belgium",
    "Salzburg, Austria",
    "Copenhagen, Denmark",
]
SEED = 42
N_PER_CITY = 20
OUTPUT_DIR = "results"
MODEL_PATH = "models/ppo_walking_v2"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    model: MaskablePPO, env: WalkingEnvV2, case: dict
) -> dict:
    env._set_active_city(case["city"])
    obs, _ = env.reset(options={"target_duration": case["target_duration"]})
    done = False
    while not done:
        masks = env.action_masks()
        action, _ = model.predict(
            obs[np.newaxis], action_masks=masks[np.newaxis], deterministic=True
        )
        obs, _, terminated, truncated, info = env.step(int(action[0]))
        done = terminated or truncated

    m = compute_loop_metrics(
        env.G,
        info["episode_path"],
        case["target_duration"],
        info["current_duration"],
        episode_pois=env.episode_pois,
    )
    m["city"] = info["current_city"]
    return m


# ---------------------------------------------------------------------------
# Hero plot
# ---------------------------------------------------------------------------

def _save_hero_plot(df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.bar(
        range(1, len(df) + 1), df["pct_error"] * 100,
        color="steelblue", edgecolor="none", alpha=0.7
    )
    ax.axhline(10, color="r", linestyle="--", label="10% tol.")
    mean_err = df["pct_error"].mean() * 100
    ax.axhline(mean_err, color="g", linestyle="-", label=f"Mean {mean_err:.1f}%")
    ax.set_title("% error per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("% error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    bucket_order = ["10-30", "30-60", "60-90", "90-120"]
    bucket_data = [
        df[df["target_bucket"] == b]["pct_error"].values * 100 for b in bucket_order
    ]
    bp = ax.boxplot(bucket_data, labels=bucket_order, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.5)
    ax.axhline(10, color="r", linestyle="--", label="10% tol.")
    ax.set_title("Error by target bucket")
    ax.set_xlabel("Target duration (min)")
    ax.set_ylabel("% error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    city_stats = (
        df.groupby("city")
        .agg(mean_error=("pct_error", lambda x: x.mean() * 100))
        .reset_index()
    )
    x = np.arange(len(city_stats))
    ax.bar(x, city_stats["mean_error"], color="C0", edgecolor="black", alpha=0.8)
    ax.axhline(10, color="r", linestyle="--", label="10% tol.")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.split(",")[0] for c in city_stats["city"]], rotation=30, ha="right"
    )
    ax.set_title("Per-city mean error (zero-shot)")
    ax.set_ylabel("% error (mean)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    quality_cols = ["circularity", "compactness", "success_10"]
    quality_means = df[quality_cols].mean()
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    ax.bar(quality_cols, quality_means.values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_title("Quality metrics (mean over test set)")
    ax.set_ylabel("Score [0, 1]")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "hero_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Hero plot saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config | None = None) -> None:
    cfg = cfg or Config.default()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    model_path = MODEL_PATH
    if not Path(f"{model_path}.zip").exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH}.zip. Run `python -m walking_loops.train` first."
        )

    env = WalkingEnvV2(places=TEST_CITIES, cfg=cfg.env)
    model = MaskablePPO.load(model_path)

    test_set = make_deterministic_test_set(
        cities=TEST_CITIES, n_per_city=N_PER_CITY, seed=SEED
    )
    print(f"Running {len(test_set)} deterministic evaluation episodes …\n")

    all_metrics: list[dict] = []
    for i, case in enumerate(test_set):
        m = _run_episode(model, env, case)
        all_metrics.append(m)
        sign = "v" if m["success_10"] else "x"
        print(
            f"  [{i+1:03d}] {sign} {m['city']:30s} "
            f"target={m['target_duration']:5.1f}  "
            f"actual={m['actual_duration']:5.1f}  "
            f"err={m['pct_error']*100:5.1f}%  "
            f"circ={m['circularity']:.2f}"
        )

    env.close()

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(OUTPUT_DIR, "test_metrics.csv"), index=False)

    summary = (
        df.groupby(["city", "target_bucket"])
        .agg(
            pct_error_mean=("pct_error", lambda x: x.mean() * 100),
            pct_error_std=("pct_error", lambda x: x.std() * 100),
            success_rate_10=("success_10", "mean"),
            circularity_mean=("circularity", "mean"),
            edge_revisit_mean=("edge_revisit_ratio", "mean"),
            n_episodes=("pct_error", "count"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(OUTPUT_DIR, "test_summary.csv"), index=False)

    _save_hero_plot(df, OUTPUT_DIR)

    # Console report
    print("\n" + "=" * 60)
    print("FINAL EVALUATION REPORT — WalkingEnvV2")
    print("=" * 60)
    print(f"  Episodes          : {len(df)}")
    print(f"  Mean % error      : {df['pct_error'].mean()*100:.2f}%")
    print(f"  Median % error    : {df['pct_error'].median()*100:.2f}%")
    print(f"  Success <=10%     : {df['success_10'].mean()*100:.1f}%")
    print(f"  Success <=20%     : {df['success_20'].mean()*100:.1f}%")
    print(f"  Mean circularity  : {df['circularity'].mean():.3f}")
    print(f"  Mean edge-revisit : {df['edge_revisit_ratio'].mean():.3f}")
    print()

    city_grp = df.groupby("city").agg(
        err=("pct_error", lambda x: x.mean() * 100),
        sr=("success_10", "mean"),
        n=("pct_error", "count"),
    )
    for city, row in city_grp.iterrows():
        print(f"  {city:35s}  err={row['err']:.1f}%  sr={row['sr']*100:.0f}%  n={int(row['n'])}")
    print("=" * 60)

    stats_path = os.path.join(OUTPUT_DIR, "statistics.txt")
    with open(stats_path, "w") as f:
        f.write("=== EVALUATION STATISTICS — WalkingEnvV2 ===\n")
        f.write(f"Seed           : {SEED}\n")
        f.write(f"Episodes       : {len(df)}\n")
        f.write(f"Mean error     : {df['pct_error'].mean()*100:.2f}%\n")
        f.write(f"Median error   : {df['pct_error'].median()*100:.2f}%\n")
        f.write(f"Success <=10%  : {df['success_10'].mean()*100:.1f}%\n")
        f.write(f"Circularity    : {df['circularity'].mean():.3f}\n")
        f.write(f"Edge revisit   : {df['edge_revisit_ratio'].mean():.3f}\n\n")
        f.write(city_grp.to_string())
    print(f"Statistics -> {stats_path}")


if __name__ == "__main__":
    main()
