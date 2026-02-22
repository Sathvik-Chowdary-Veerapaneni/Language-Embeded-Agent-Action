# LEAA v1 Training — Results & Insights

## Training Configuration
- **Algorithm**: PPO (Stable-Baselines3)
- **Device**: CPU (M1 MacBook Pro, 8 cores)
- **Parallel Envs**: 8 (SubprocVecEnv)
- **Total Timesteps**: 2,000,000
- **Network**: MLP [256, 256, 128] with Tanh activations
- **Hyperparameters**:
  - `gamma=1.0` (single-step episodes)
  - `ent_coef=0.03`
  - `learning_rate=1e-4`
  - `batch_size=256`
  - `n_steps=2048`
  - `clip_range=0.2`
- **Training Time**: ~125 minutes (2 hours 5 min)
- **Total Episodes**: 1,694,312

## Curriculum Progression

| Stage | Name | Distance | Radius | Passed? | Peak Success |
|-------|------|----------|--------|---------|-------------|
| 0 | static_close | 3-10m | 2.0m | ✅ 90% | 90.0% |
| 1 | static_medium | 8-20m | 1.5m | ✅ 85% | 85%+ |
| 2 | static_far | 15-50m | 1.5m | ✅ 85.2% | 85.2% |
| 3 | moving_slow | 10-40m | 1.5m | ⚠️ 2 steps only | 10% |
| 4 | wind | — | — | ❌ Not reached | — |
| 5 | full_dynamic | — | — | ❌ Not reached | — |

## Evaluation Results (with VecNormalize)

| Stage | Hit Rate | Bullseye | Inner | Outer | Avg Dist |
|-------|----------|----------|-------|-------|----------|
| static_close | 63.5% | 0 | 0 | 127 | 2.000 |
| static_medium | 73.5% | 0 | 0 | 147 | 1.500 |
| static_far | 23.0% | 0 | 0 | 46 | 1.500 |
| moving_slow | 1.5% | 0 | 0 | 3 | 1.500 |
| wind | 5.5% | 0 | 0 | 11 | 1.000 |
| full_dynamic | 1.5% | 0 | 0 | 3 | 0.750 |
| **Overall** | **28.1%** | **0** | **0** | **337** | — |

## Key Issues Identified

### 1. Exploration Collapsed Too Early
- **Symptom**: `std` dropped from 0.96 → 0.097 during training
- **Impact**: Agent locked into one aiming strategy, stopped finding better solutions
- **Root Cause**: `ent_coef=0.03` was still too low for 2M steps
- **Fix for v2**: Increase to `ent_coef=0.05-0.08`, add entropy coefficient schedule

### 2. Zero Bullseyes — All Hits Are Edge-Only
- **Symptom**: Every hit has `distance_from_center = target_radius` (barely clipping the edge)
- **Impact**: No precision, just "close enough to graze"
- **Root Cause**: Reward tiers (100/50/25) create a flat gradient within each zone — the agent has no incentive to aim for center once it can hit the outer ring
- **Fix for v2**: Add continuous distance-based bonus within each tier

### 3. VecNormalize Stats Mismatch During Evaluation
- **Symptom**: `static_close` eval (63.5%) was WORSE than training (90%)
- **Root Cause**: Stats evolved through stages 0→1→2→3, final stats optimized for later stages
- **Fix for v2**: Save per-stage VecNormalize stats, use stage-matched stats during eval

### 4. Budget Ran Out Before Moving/Wind Stages
- **Symptom**: Only 2 timesteps on `moving_slow`, never reached `wind` or `full_dynamic`
- **Root Cause**: 2M timesteps split across 6 stages, static stages consumed most budget
- **Fix for v2**: 5M+ timesteps, or dynamically allocate more budget to harder stages

### 5. Explained Variance Plateau
- **Symptom**: `explained_variance` peaked at ~30%, never climbed higher
- **Impact**: Value function not accurately predicting returns
- **Fix for v2**: Consider separate value network architecture, or normalize advantages differently

## Trajectory Plots
See `eval_plots/` directory for 3D trajectory visualizations of each stage

## What Worked Well ✅
- **No NaN crashes** — the per-stage `model.learn()` fix eliminated the stage transition bug
- **Seeded wind RNG** — deterministic rewards improved learning stability
- **Curriculum progression** — agent successfully advanced through 3 static stages
- **Target prediction** — moving target position prediction at impact time implemented correctly
- **Checkpoint system** — reliable save/resume with VecNormalize stats

## v2 Training Plan
See `results/v2/PLAN.md` for the v2 improvement plan
