"""
walking-loops — Reinforcement Learning for walking loop generation.

An attention-based PPO agent that builds circular pedestrian routes on real
OpenStreetMap city graphs, hitting a user-specified target duration.
"""

__version__ = "2.0.0"

__all__ = ["config", "env", "policy", "train", "eval", "graph_utils", "metrics"]
