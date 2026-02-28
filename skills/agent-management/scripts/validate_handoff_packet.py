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


def validate_packet(data: dict, index: int | None = None) -> list[str]:
    prefix = f"packet {index}: " if index is not None else ""
    errors = []
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        errors.append(f"{prefix}missing keys: {', '.join(missing)}")

    if not isinstance(data.get("next_actions"), list) or not data.get("next_actions"):
        errors.append(f"{prefix}next_actions must be a non-empty list")
    if not isinstance(data.get("reference_subset_recommended"), list) or not data.get("reference_subset_recommended"):
        errors.append(f"{prefix}reference_subset_recommended must be a non-empty list")
    
    return errors


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python validate_handoff_packet.py <packet.json> [--chain]")
        return 2
    
    is_chain = "--chain" in sys.argv
    path_arg = sys.argv[1] if sys.argv[1] != "--chain" else sys.argv[2]
    
    path = Path(path_arg)
    if not path.exists():
        print(f"missing file: {path}")
        return 2

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"invalid JSON: {exc}")
        return 2

    all_errors = []
    if is_chain:
        if not isinstance(data, list):
            print("invalid chain: expected a list of packets")
            return 1
        
        last_session_id = None
        for i, packet in enumerate(data):
            all_errors.extend(validate_packet(packet, i))
            
            # Chain integrity check
            if i > 0 and last_session_id:
                parent_id = packet.get("parent_session_id")
                if parent_id != last_session_id:
                    all_errors.append(f"packet {i}: parent_session_id '{parent_id}' does not match predecessor session_id '{last_session_id}'")
            
            last_session_id = packet.get("session_id")
    else:
        if not isinstance(data, dict):
            print("invalid packet: expected a dictionary")
            return 1
        all_errors.extend(validate_packet(data))

    if all_errors:
        for err in all_errors:
            print(f"error: {err}")
        return 1

    print(f"handoff {'chain' if is_chain else 'packet'}: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
