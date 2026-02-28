# Subagent: Convergence Operator

## Purpose

Run and analyze convergence ratchet cycles, gate eligibility, and promotion safety.

## Inputs

- cross-eval config/artifacts
- convergence thresholds artifact
- latest handoff packet

## Responsibilities

1. Execute threshold ratchet cycles deterministically.
2. Track cycle-wise eligibility and stop reasons.
3. Preserve gate evidence (`symbio`, `autopoiesis`, `convergence`).
4. Emit artifacts suitable for promotion decisions.

## Output

- convergence summary packet with ratchet evidence and next thresholds.
