from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_KEYS = [
    "session_id",
    "parent_session_id",
    "agent_role",
    "timestamp_utc",
    "state",
    "goal",
    "next_actions",
    "return_trigger",
    "return_trigger_status",
    "lesson_tags",
    "scenario_class",
    "reference_subset_recommended",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python validate_handoff_packet.py <packet.json>")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"missing file: {path}")
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        print(f"invalid packet: missing keys: {', '.join(missing)}")
        return 1

    if not isinstance(data.get("next_actions"), list) or not data["next_actions"]:
        print("invalid packet: next_actions must be a non-empty list")
        return 1
    if not isinstance(data.get("reference_subset_recommended"), list) or not data["reference_subset_recommended"]:
        print("invalid packet: reference_subset_recommended must be a non-empty list")
        return 1

    print("handoff packet: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
