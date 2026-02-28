# Handoff Playbook

Use this with:
- `AGENTS.md`
- `.agents/handoff-packet-template.md`
- `.agents/handoff-protocol.md`
- `.agents/return-protocol.md`

## Minimal Handoff Procedure

1. Confirm lifecycle state is `VALIDATION_LOCKED`.
2. Build packet from template.
3. Populate deterministic next actions.
4. Populate `reference_subset_recommended` with minimal files only.
5. Validate packet using `scripts/validate_handoff_packet.py`.
6. Move to `HANDOFF_READY`.

## Minimal Return Procedure

1. Confirm return trigger status (`met` or `blocked`).
2. Emit delta packet with parent linkage.
3. Include completion status, residual risk, and next continuation branch.
4. Move to `RETURN_READY`.

## Context Overhead Guidance

1. Do not attach full experiment log by default.
2. Prefer compact lesson references (`session-lessons-latest`, `lessons-common`, `lessons-unique`).
3. Keep packet concise but decision-complete.
