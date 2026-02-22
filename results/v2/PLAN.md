# LEAA v2 Training Plan

## Lessons from v1

| Problem | Root Cause | v2 Fix |
|---------|-----------|--------|
| 0 bullseyes | Flat reward within tiers | Continuous distance bonus |
| std → 0.097 | ent_coef too low | ent_coef=0.05 + schedule |
| Only 3/6 stages trained | 2M steps not enough | 5M steps + dynamic budget |
| Eval worse than training | VecNormalize stats mismatch | Per-stage eval stats |
| Explained variance ~30% | Shared network bottleneck | Larger value head |

## v2 Hyperparameter Changes

```python
# v1 → v2
gamma       = 1.0        # unchanged (single-step)
ent_coef    = 0.05       # was 0.03 — slower exploration decay
learning_rate = 3e-4     # was 1e-4 — faster learning
n_steps     = 2048       # unchanged
batch_size  = 512        # was 256 — larger batches
clip_range  = 0.2        # unchanged
num_envs    = 12         # was 8 — 50% faster data collection
timesteps   = 5_000_000  # was 2M — 2.5x more training
```

## v2 Reward Shaping Fix

The v1 reward gives the same 25 points for hitting the edge as hitting close to center within the outer ring. v2 adds a continuous bonus:

```python
# v2 reward = tier_reward + precision_bonus
if hit:
    tier = bullseye(100) or inner(50) or outer(25)
    precision_bonus = 10.0 * (1.0 - dist_from_center / radius)
    reward = tier + precision_bonus
```

This gives the agent a gradient signal: hitting closer to center always pays more

## v2 Network Architecture

```python
# v1: shared [256, 256, 128]
# v2: separate policy and value networks
policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256],    # Policy network (actor)
        vf=[256, 256],    # Value network (critic) — independent
    ),
)
```

Separate networks let the value function learn independently without competing with the policy for representation capacity

## v2 Entropy Schedule

```python
# Linear decay: 0.05 → 0.01 over training
def ent_schedule(progress):
    return 0.05 - 0.04 * progress  # progress goes 0→1
```

Explores broadly early, focuses precisely later

## v2 Dynamic Stage Budget

Instead of dividing timesteps evenly, give harder stages more time:

```python
stage_weights = [1, 1, 2, 3, 4, 5]  # harder stages get more budget
```

## v2 Evaluation Fix

Load per-stage VecNormalize stats during evaluation to match training conditions exactly
