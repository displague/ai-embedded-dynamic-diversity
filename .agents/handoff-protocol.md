# Handoff Protocol

Strict lifecycle for successive agent operations in this repository.

## States

1. `DISCOVERY_LOCKED`
2. `PLAN_LOCKED`
3. `EXECUTION_ACTIVE`
4. `VALIDATION_LOCKED`
5. `HANDOFF_READY`
6. `HANDOFF_ACCEPTED`
7. `RETURN_READY`
8. `RETURN_ACCEPTED`

## Transition Requirements

### `DISCOVERY_LOCKED -> PLAN_LOCKED`
- Current repo state and constraints verified.
- Target files/interfaces identified.

### `PLAN_LOCKED -> EXECUTION_ACTIVE`
- Plan is decision-complete and scoped.

### `EXECUTION_ACTIVE -> VALIDATION_LOCKED`
- Work implemented.
- Relevant validation evidence collected.

### `VALIDATION_LOCKED -> HANDOFF_READY`
- Handoff packet complete.
- Next actions deterministic.

### `HANDOFF_READY -> HANDOFF_ACCEPTED`
- Successor confirms packet completeness and scope.

### `HANDOFF_ACCEPTED -> RETURN_READY`
- Return trigger met, or blocking issue discovered.

### `RETURN_READY -> RETURN_ACCEPTED`
- Receiver validates packet and continuation path.

## Packet Chain Rules

1. Every packet references `parent_session_id`.
2. Every execution handoff includes validation evidence.
3. Every handoff includes `reference_subset_recommended`.
4. Every return includes `return_trigger_status`.
