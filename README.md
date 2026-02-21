# LEAA - Language Embedded Agent Actions

Archery agent in a 3D environment that takes natural language commands to hit flagged targets.

## POC Architecture
```
Phase 1: RL Policy (hit things with arrows — physics, wind, movement)
Phase 2: Language Grounding (LLM parses "hit yellow flag" → target spec → RL policy executes)
```


## Development Stack
- **3D Engine**: PyBullet (prototype) → Unity (production)
- **RL**: Stable-Baselines3 + PPO on MPS (Apple Silicon)
- **Language**: Claude API / Ollama (Phi-3 local)
- **Device**: M1 MacBook Pro, MPS backend
  