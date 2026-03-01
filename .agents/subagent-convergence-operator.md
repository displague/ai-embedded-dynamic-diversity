# Subagent: Convergence Operator

## Purpose

Run and analyze convergence ratchet cycles, gate eligibility, and promotion safety.

## Inputs

- cross-eval config/artifacts
- convergence thresholds artifact
- latest handoff packet

## Scripts

- `skills/ai-convergence-research/scripts/gen_v7_tape.py`: extract constructor tape from champion weights.
- `skills/ai-convergence-research/scripts/symbio_audit.py`: quantitatively audit symbiogenesis gain from coevolution logs.
- `skills/ai-convergence-research/scripts/shock_test.py`: measure model stability under volatility shocks.
- `skills/ai-convergence-research/scripts/export_int8.py`: dynamically quantize weights to INT8 TorchScript.
- `skills/ai-convergence-research/scripts/pi_bench.py`: end-to-end HIL/simulation benchmark for Pi 5 latency.

## Responsibilities

1. Execute threshold ratchet cycles deterministically.
2. Track cycle-wise eligibility and stop reasons.
3. Preserve gate evidence (`symbio`, `autopoiesis`, `convergence`).
4. Emit artifacts suitable for promotion decisions.

## Output

- convergence summary packet with ratchet evidence and next thresholds.
