# .agents Layout

Canonical in-repo agent orchestration and hand-off assets.

## Structure

- Flat files in `.agents/` using prefixes:
  - `subagent-*` role contracts (short, decision-focused)
  - `handoff-*` lifecycle/packet/return protocol
  - `skill-*` compatibility pointers

## Compatibility Strategy

Canonical role content lives in `.agents/subagent-*.md`.
Mirrored adapters are maintained for:
- `.claude/agents`
- `.opencode/agents`

Avoid symlink-only designs for Windows portability.

## Startup Order For Successor Agents

1. Read `AGENTS.md`.
2. Read assigned role file (`.agents/subagent-*.md`).
3. Read relevant skill `SKILL.md`.
4. Load only references listed in the hand-off packet field:
   `reference_subset_recommended`.
