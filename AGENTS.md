# AGENTS

Repository-level agent contract for `ai-embedded-dynamic-diversity`.

## Mission

Advance convergence across:
- diversity (embodiments and scenario spread)
- hardiness (stress/noise/disturbance robustness)
- replication/emergence (prelife/symbiogenesis proxies)
- embodied autopoietic computation

All work should preserve anonymous-channel adaptation and cross-embodiment transfer goals.

## Source Of Truth Order

When guidance conflicts, use this precedence:
1. `AGENTS.md` (this file)
2. Active hand-off packet for current session (from `.agents/handoff-packet-template.md`)
3. Role contract (`.agents/subagents/*.md`)
4. Skill definition (`skills/*/SKILL.md`)
5. Deep references (`skills/*/references/*`)

## Context Budget Rules

1. Load small definition files first.
2. Load heavy references only when needed.
3. Do not bulk-ingest long logs by default.
4. Prefer targeted reference subsets listed in hand-off packets.

## Promotion And Convergence Discipline

1. Champion promotion must be based on cross-eval artifacts, not training fitness alone.
2. Prefer high-repeat validation for promotion decisions.
3. Keep ratchet thresholds explicit and versioned in artifacts.
4. Include visual evidence (storyboard/compare gifs) for embodied behavior changes.

## Commit And Change Discipline

1. Use detailed, scoped commits.
2. Keep docs and workflow instructions synchronized with code changes.
3. Update hand-off assets when changing convergence/viz/reporting workflows.

## Strict Lifecycle States

Work must move through these states:
1. `DISCOVERY_LOCKED`
2. `PLAN_LOCKED`
3. `EXECUTION_ACTIVE`
4. `VALIDATION_LOCKED`
5. `HANDOFF_READY`
6. `HANDOFF_ACCEPTED`
7. `RETURN_READY`
8. `RETURN_ACCEPTED`

See `.agents/handoff-protocol.md` for transition requirements.

## Required Hand-Off Packet

Every transfer between agents requires a packet with:
- session lineage metadata
- goal/scope boundaries
- changed files and rationale
- validation evidence
- unresolved risks/assumptions
- deterministic next actions
- return trigger
- recommended minimal reference subset

Use `.agents/handoff-packet-template.md`.

## Cross-Platform Agent Compatibility

Canonical in this repo:
- `.agents/subagents/*.md`
- `.agents/handoff-protocol.md`
- `.agents/handoff-packet-template.md`
- `.agents/return-protocol.md`
- `skills/*`

Compatibility mirrors:
- `.claude/agents/*.md`
- `.opencode/agents/*.md`

Mirrors should remain concise and aligned with canonical role responsibilities.
