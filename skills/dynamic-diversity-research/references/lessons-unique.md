# Lessons: Unique / Edge Cases

## U1: Filesystem ACL differences may constrain preferred layout

- situation: dot-directory nesting can be blocked by environment ACL policy.
- action: preserve canonical semantics using flat prefixed files when nested structure is unavailable.
- outcome: compatibility preserved without blocking handoff readiness.
- transferability: medium.
- tags: `unique`, `handoff`, `failure-mode`.

## U2: CUDA environment drift can invalidate long experiment assumptions

- situation: runtime can silently shift from CUDA to CPU in some workflows.
- action: enforce strict device checks and record execution path in artifacts.
- outcome: avoids false confidence from unintended runtime.
- transferability: high.
- tags: `unique`, `hardware`, `failure-mode`.

## U3: Ratchet cycle frontier is a meaningful artifact

- situation: cycle 1 passes but cycle 2 fails eligibility.
- action: treat fail boundary as convergence frontier and store stop reason explicitly.
- outcome: clearer next-threshold planning.
- transferability: medium.
- tags: `unique`, `convergence`.
