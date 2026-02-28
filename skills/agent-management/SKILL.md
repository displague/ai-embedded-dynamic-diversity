---
name: agent-management
description: Devex and management workflows for multi-agent handoff, continuity, and contract synchronization.
---

# Agent Management Skill

## Use This Skill When

1. Preparing or validating multi-agent handoff packets and return-ready continuity updates.
2. Synchronizing canonical role contracts with ecosystem mirrors (.claude/.opencode).
3. Auditing repository-level agent policy (AGENTS.md) and protocol drift.

## Core Invariants

1. Preserve packet chain integrity (`parent_session_id`).
2. Adhere to strict handoff/return lifecycle states (AGENTS.md).
3. Minimize successor context load using targeted reference subsets.

## Scripts And Assets

- `scripts/check_agent_drift.py`: verify canonical contracts and mirrors are synchronized.
- `scripts/gen_reference_subset.py`: generate minimal reference subsets for handoff packets based on tags.
- `scripts/validate_handoff_packet.py`: validate required packet fields.
- `assets/handoff-packet-example.json`: starting packet example.
