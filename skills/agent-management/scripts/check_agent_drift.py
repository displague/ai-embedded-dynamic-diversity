import os
import sys
from pathlib import Path

def check_agent_drift():
    root = Path(".")
    agents_dir = root / ".agents"
    claude_dir = root / ".claude" / "agents"
    opencode_dir = root / ".opencode" / "agents"

    if not agents_dir.exists():
        print("Error: .agents directory not found.")
        return 1

    canonical_agents = list(agents_dir.glob("subagent-*.md"))
    canonical_names = [f.name.replace("subagent-", "").replace(".md", "") for f in canonical_agents]

    drift_found = False

    # Check for missing mirrors
    for name in canonical_names:
        claude_mirror = claude_dir / f"{name}.md"
        if not claude_mirror.exists():
            print(f"Drift: Missing Claude mirror for {name} at {claude_mirror}")
            drift_found = True
        
        opencode_mirror = opencode_dir / f"{name}.md"
        if not opencode_mirror.exists():
            print(f"Drift: Missing OpenCode mirror for {name} at {opencode_mirror}")
            drift_found = True

    # Check for mirrors without canonical sources
    if claude_dir.exists():
        for mirror in claude_dir.glob("*.md"):
            if mirror.name.replace(".md", "") not in canonical_names:
                print(f"Drift: Claude mirror {mirror.name} has no canonical source in .agents")
                drift_found = True

    if opencode_dir.exists():
        for mirror in opencode_dir.glob("*.md"):
            if mirror.name.replace(".md", "") not in canonical_names:
                print(f"Drift: OpenCode mirror {mirror.name} has no canonical source in .agents")
                drift_found = True

    # Check for content drift in mirrors (ensure they reference the canonical source)
    for name in canonical_names:
        canonical_file = f"subagent-{name}.md"
        
        claude_mirror = claude_dir / f"{name}.md"
        if claude_mirror.exists():
            content = claude_mirror.read_text()
            if canonical_file not in content:
                print(f"Drift: Claude mirror {claude_mirror.name} does not reference {canonical_file}")
                drift_found = True

        opencode_mirror = opencode_dir / f"{name}.md"
        if opencode_mirror.exists():
            content = opencode_mirror.read_text()
            if canonical_file not in content:
                print(f"Drift: OpenCode mirror {opencode_mirror.name} does not reference {canonical_file}")
                drift_found = True

    if drift_found:
        print("\nAgent continuity drift detected.")
        return 1
    else:
        print("No agent continuity drift detected.")
        return 0

if __name__ == "__main__":
    sys.exit(check_agent_drift())
