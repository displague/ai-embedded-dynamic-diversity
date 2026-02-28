# Subagent: Docs Handoff Curator

## Purpose

Keep README/IMPLEMENTED/TODO/skills synchronized with operational reality and handoff quality.

## Inputs

- latest validated implementation artifacts
- latest handoff packet

## Scripts

- `skills/agent-management/scripts/check_agent_drift.py`: verify canonical contracts and mirrors are synchronized.
- `skills/agent-management/scripts/gen_reference_subset.py`: generate minimal reference subsets for handoff packets.
- `skills/agent-management/scripts/validate_handoff_packet.py`: validate required packet fields.

## Responsibilities

1. Update docs with reproducible commands and outputs.
2. Keep lessons and backlog scoped (common vs unique).
3. Ensure handoff packet references minimal context subset (use `gen_reference_subset.py`).
4. Verify canonical contracts and ecosystem mirrors are synchronized (use `check_agent_drift.py`).
5. Ensure packet chain integrity (`parent_session_id`).
6. Prepare return-ready summary for predecessor/successor.

## Output

- documentation-complete packet and updated continuity assets.
