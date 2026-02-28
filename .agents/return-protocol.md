# Return Protocol

Return is a first-class lifecycle event, not an ad-hoc note.

## Return Preconditions

1. Current state is `HANDOFF_ACCEPTED`.
2. Return trigger is `met` or `blocked`.
3. Delta packet is prepared from the latest parent packet.

## Required Return Payload

1. Completed items (with file and artifact references).
2. Remaining items (explicitly ordered).
3. Behavioral/regression status for convergence gates.
4. Context minimization report:
- what was loaded
- what was intentionally not loaded
5. Recommended next continuation branch.

## Return Acceptance Criteria

1. Receiver can continue with only:
- packet
- `reference_subset_recommended`
- role + skill definitions
2. No hidden dependency on untracked chat history.
