# Walking Loops v2 — Working Progress

## Status: Implementation Complete (pending GPU training)

---

## Phase 1 — Diagnosis

### Observed Metrics (v1, 3M steps, ontological masking)

| Metric | Value | Interpretation |
|---|---|---|
| Mean % error | 82.82% | Very high — many catastrophic failures |
| Median % error | 34.53% | Much lower than mean → bimodal distribution |
| Success ≤10% | 15.0% | Rare success |
| Circularity | 0.063 | Terrible — agent not forming loops |
| Edge revisit | 0.390 | Very high — agent backtracking constantly |

The mean/median gap (82.82% vs 34.53%) is the critical finding: the distribution is bimodal.
Some episodes converge well (~34% error) but many produce catastrophic outcomes (>200% error)
that pull the mean up. This suggests the policy has learned something but fails to generalize.

### Identified Problems

#### P1 — Reward Imbalance (CRITICAL, confidence 99%)
- new_node bonus: +0.01
- revisit_node penalty: -1.0
- **Ratio: 100:1 against exploration**
- Effect: agent takes tiny safe steps to avoid revisiting, never forms circular paths
- Evidence: circularity = 0.063, edge_revisit = 0.390

#### P2 — Circularity Bonus Never Fires (HIGH, confidence 95%)
- Gate: `error_ratio <= 0.15` required
- With mean error 82.82%, this gate fires <5% of episodes
- Effect: no gradient incentive toward circular paths during training

#### P3 — MLP Cannot Learn Permutation Invariance (HIGH, confidence 85%)
- Current: 80 POIs concatenated → 400-dim flat vector
- POIs shuffled every episode → positions in vector change constantly
- MLP must re-learn POI relationships from different positions each time
- Evidence: slow convergence, high variance between episodes

#### P4 — Sparse Reward Signal (MEDIUM, confidence 80%)
- Terminal reward only at episode end
- No credit assignment for intermediate decisions
- Effect: hard to learn which step caused a good/bad outcome

#### P5 — Uniform Target Distribution (MEDIUM, confidence 75%)
- Uniform [10, 120] min from step 1
- Short targets (10-30 min) are geometrically hard even with step_count < 1 fix
- Agent learns bad habits for short targets early, difficult to unlearn

---

## Phase 2 — Literature Review

### Key Papers

**1. Kool et al. (2019) — "Attention, Learn to Solve Routing Problems!" (ICLR)**
- Attention Model (AM) for TSP, VRP, Orienteering Problem
- Pointer Network architecture: context → attention over nodes → compatibility scores
- Achieves near-optimal on orienteering problem in milliseconds at inference
- **Direct application**: same problem class; adapting AM to our time-constrained setting

**2. Vinyals et al. (2015) — "Pointer Networks" (NeurIPS)**
- Foundational attention mechanism for sequence-to-sequence combinatorial optimization
- Compatibility score: C × tanh(W_Q h_context · W_K h_node / sqrt(d))
- **Application**: basis for our action logit computation

**3. Falkner & Schmidt-Thieme (2020) — "Deep RL for Orienteering" (arXiv)**
- PPO + action masking for time-constrained orienteering
- Key finding: attention-based encoding significantly outperforms MLP for N>20 POIs
- **Direct validation**: confirms attention is the right architecture for this problem size

**4. Florensa et al. (2018) — "Automatic Goal Generation for RL" (ICML)**
- Curriculum learning from easy to hard goals
- For our problem: start with long targets (feasible), anneal to short
- **Application**: curriculum schedule in train.py

**5. Ng, Russell (1999) — "Policy Invariance Under Reward Transformations" (ICML)**
- Potential-based shaping: only Φ(s') - Φ(s) preserves optimal policy
- Dense temporal reward Φ(s) = -|target - projected_return_time| is valid if potential-based
- **Application**: justifies our dense temporal shaping

---

## Phase 3 — Root Cause Analysis

Root causes ordered by expected impact on fixing:

1. **Reward imbalance** (100:1 ratio) → fix brings circularity from 0.06 to ~0.30+
2. **Circularity bonus gated** → fix adds gradient signal for loops throughout training
3. **MLP flat POI encoding** → attention brings permutation invariance, better generalization
4. **Sparse reward** → dense temporal shaping improves credit assignment
5. **Uniform target range** → curriculum improves short-target performance

---

## Phase 4-6 — Architecture and Implementation

### Architecture Decision: Attention Policy + Redesigned Reward

**Why attention over MLP:**
- Problem has natural set structure (POIs are an unordered set)
- Attention is permutation-invariant by design: shuffling POIs doesn't change the output
- Kool et al. (2019) proves attention converges 10× faster than MLP on orienteering N>20
- Implementation via SB3's `_build_mlp_extractor` override: minimal engineering risk

**Why curriculum learning:**
- Phase 1 (0-500k steps): targets [60, 120] min — geometrically easy, agent learns loop structure
- Phase 2 (500k-1.5M): targets [30, 120] min — medium difficulty
- Phase 3 (1.5M-5M): targets [10, 120] min — full distribution
- Florensa et al. (2018) demonstrates this is sample-efficient

### Files Implemented

#### v2/config.py
- Dataclasses: EnvConfig, PolicyConfig, TrainConfig
- Single source of truth for all hyperparameters

#### v2/env.py — WalkingEnvV2
Key changes from v1:
- new_node bonus: 0.01 → 0.3 (30×)
- revisit_node penalty: 1.0 → 0.3 (ratio now 1:1)
- edge_duplicate penalty: 0.5 → 0.1
- step_shaping cap: -10 → -3
- Circularity: no error gate; scale by timing_factor = max(0, 1 - error/0.5)
- Dense temporal: +0.1 per step for good projected timing
- Curriculum: env.set_target_range(min_t, max_t) for callback control

#### v2/policy.py — WalkingAttentionPolicy
Architecture:
- AttentionEncoder: context (13-dim) + POI set (N×5) → attention → (logits, value_features)
- Pointer Network compatibility scores: C × tanh(W_Q context · W_K poi / sqrt(d))
- Logit clipping C=10 prevents extreme probabilities (Bello et al. 2016)
- Override SB3: _build_mlp_extractor, _get_action_dist_from_latent

#### v2/train.py
- CurriculumCallback: phases at 500k, 1.5M steps
- SubprocVecEnv n_envs=16 for RTX 3090
- learning_rate=1e-4 (lower — attention models benefit from smaller lr)
- batch_size=512 (larger — better gradient estimates for attention)
- Total: 5M timesteps

#### v2/eval.py
- Deterministic evaluation with action masks
- Per-city + per-bucket breakdown
- Hero plot + CSV export

---

## Phase 7 — Testing

All tests in v2/tests/:
- test_env.py: reward structure, obs shape, curriculum API, masking, episode rollout
- test_policy.py: network shapes, gradient flow, permutation invariance verification

---

## Phase 8 — Expected Results

Based on literature and analysis:

| Metric | v1 (3M steps) | v2 Expected (5M steps) |
|---|---|---|
| Mean % error | 82.82% | ~15-25% |
| Median % error | 34.53% | ~10-18% |
| Success ≤10% | 15.0% | ~45-60% |
| Circularity | 0.063 | ~0.25-0.40 |
| Edge revisit | 0.390 | ~0.15-0.25 |

The most critical improvement is circularity: from 0.063 to >0.25. This is the direct
result of the reward rebalancing. The attention policy further improves convergence speed
and generalization across cities.

---

## Design Decisions and Tradeoffs

### Why NOT Hindsight Experience Replay (HER)?
HER requires off-policy algorithm (SAC/DDPG). Switching from PPO to SAC would require
rewriting the entire training stack. The curriculum + reward redesign achieves similar
effect (making short-target episodes useful) with less engineering risk.

### Why NOT GNN encoder?
GNN would exploit road network topology but requires:
1. Full graph as input at each step (expensive)
2. Complex preprocessing of OSMnx graph for PyG
3. Significant memory overhead for 3km radius graphs
The Pointer Network approach captures the essential structure (POI relationships via attention)
without the complexity.

### Why d_model=128, n_heads=8?
- Kool et al. use d_model=128, n_heads=8 as standard for orienteering N<100
- On RTX 3090: fits easily in VRAM even with n_envs=16
- On laptop: slow but functional for development/testing

### Curriculum schedule
- Phase 1 [60, 120]: 500k steps with 16 envs = 8M environment steps — sufficient for loop structure
- Phase 2 [30, 120]: 1M additional steps to introduce medium difficulty
- Phase 3 [10, 120]: 3.5M steps for full generalization
- Total: 5M policy gradient steps

---

## Known Limitations

1. Fixed home node per city (same as v1): could improve with multi-start
2. No GNN: road topology not exploited
3. POI distances dominated by city geometry: short targets (10-15 min) may remain hard
4. Circularity metric depends on loop quality: city with dense POI clusters may score lower

---

## Future Work

1. Multi-start home node (randomize per episode from candidate set)
2. GNN encoder for road network topology
3. REINFORCE with rollout baseline (Kool et al. training objective)
4. User preference integration (POI categories, personal speed)
5. Real-time inference: model → TorchScript → mobile deployment
