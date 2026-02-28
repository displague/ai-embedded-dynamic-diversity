# Lessons: Common Patterns

## C1: Promotion evidence should come from cross-eval artifacts

- situation: training fitness and transfer quality diverge.
- action: rank and promote using cross-eval outputs plus gates.
- outcome: more stable champion decisions.
- transferability: high.
- tags: `common`, `convergence`, `promotion-gates`.

## C2: High-repeat evaluation is required before champion promotion

- situation: low-repeat wins can regress under harder repeats.
- action: validate with higher `runs_per_combo`.
- outcome: fewer fragile promotions.
- transferability: high.
- tags: `common`, `convergence`.

## C3: Visual diagnostics accelerate root-cause analysis

- situation: scalar metrics hide adaptation timing behavior.
- action: generate compare/storyboard gifs under matched schedules.
- outcome: clearer remap shock and recovery interpretation.
- transferability: high.
- tags: `common`, `viz`.

## C4: Keep skill definitions thin, references deep

- situation: successors overconsume context and lose velocity.
- action: store compact workflow in `SKILL.md`, deep context in references.
- outcome: faster, cleaner successive handoffs.
- transferability: high.
- tags: `common`, `handoff`.
