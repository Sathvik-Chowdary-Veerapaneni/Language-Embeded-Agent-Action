**LEAA — Full Build Phases**

**Phase 1: Archery Physics Engine (Python/NumPy)**
- Custom ballistic solver — gravity, air drag, wind, arrow types, parabolic trajectories
- Gymnasium-compatible environment
- Static targets first, validate physics accuracy

**Phase 1.1: RL Training — Static Targets**
- PPO agent learns to aim and shoot stationary targets
- Curriculum: close range → far range → variable positions
- Goal: 95%+ hit rate on static targets

**Phase 1.2: RL Training — Dynamic Targets**
- Moving targets with varying speeds and directions
- Wind perturbation (random gusts, steady crosswinds)
- Agent-in-motion shooting
- Goal: 85%+ hit rate across all conditions

**Phase 1.3: Multi-Target Scene**
- 5 objects with different flag colors (red, blue, yellow, green, white)
- Scene registry — each object has ID, color, position, velocity
- Agent selects and hits a specific target by ID

**Phase 1.5: Unity Integration**
- Build 3D archery range in Unity
- Replicate physics equations in C#
- Export trained model to ONNX, run inference in Unity
- Visual validation — does the arrow fly the same way?

**Phase 2: Language Grounding Layer**
- LLM receives natural language prompt ("hit the yellow one")
- Scene graph query — maps language to object registry
- Outputs structured target spec → feeds to RL policy
- Claude API for complex commands, local Phi-3 for low-latency simple ones

**Phase 2.1: Complex Language Commands**
- "Hit the fastest moving target"
- "Shoot the one closest to the wall"
- "Hit yellow first, then blue"
- Sequential and conditional reasoning

**Phase 3: Full Loop**
- User speaks → LLM parses → target resolved → RL policy executes → Unity renders
- Real-time interaction in Unity scene
- Demo-ready build
