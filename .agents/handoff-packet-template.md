# Handoff Packet Template

Use JSON or Markdown+JSON block.

```json
{
  "session_id": "YYYYMMDD-HHMM-<agent>-<short-id>",
  "parent_session_id": "",
  "agent_role": "orchestrator|convergence-operator|viz-storyboard-operator|evaluation-auditor|docs-handoff-curator",
  "timestamp_utc": "2026-02-28T00:00:00Z",
  "state": "HANDOFF_READY",
  "goal": "",
  "in_scope": [],
  "out_of_scope": [],
  "constraints": [],
  "changes": [
    {
      "file": "",
      "reason": ""
    }
  ],
  "validation": {
    "commands": [],
    "key_results": [],
    "artifacts": []
  },
  "risks": [],
  "assumptions": [],
  "next_actions": [
    "Deterministic action 1",
    "Deterministic action 2"
  ],
  "return_trigger": "",
  "return_trigger_status": "pending|met|blocked",
  "lesson_tags": [],
  "scenario_class": "common|unique",
  "reusable_artifacts": [],
  "portability_notes": [],
  "reference_subset_recommended": []
}
```
