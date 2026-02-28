import sys
import json
from pathlib import Path

# Mapping of tags to reference files relative to the project root
TAG_TO_REFS = {
    "common": ["skills/dynamic-diversity-research/references/lessons-common.md"],
    "unique": ["skills/dynamic-diversity-research/references/lessons-unique.md"],
    "convergence": ["skills/dynamic-diversity-research/references/session-lessons-latest.md", "skills/dynamic-diversity-research/references/lessons-common.md"],
    "handoff": ["skills/dynamic-diversity-research/references/handoff-playbook.md"],
    "viz": ["skills/dynamic-diversity-research/references/lessons-common.md"],
    "promotion-gates": ["skills/dynamic-diversity-research/references/session-lessons-latest.md", "skills/dynamic-diversity-research/references/lessons-common.md"],
    "research": ["skills/dynamic-diversity-research/references/research-backlog.md"],
    "experiment": ["skills/dynamic-diversity-research/references/experiment-log.md"]
}

# Mapping of scenario classes to reference files
SCENARIO_TO_REFS = {
    "common": ["skills/dynamic-diversity-research/references/lessons-common.md", "skills/dynamic-diversity-research/references/session-lessons-latest.md"],
    "unique": ["skills/dynamic-diversity-research/references/lessons-unique.md", "skills/dynamic-diversity-research/references/handoff-playbook.md"]
}

def generate_subset(tags, scenario_class=None):
    refs = set()
    
    # Add mandatory/base files
    refs.add("AGENTS.md")
    refs.add("skills/dynamic-diversity-research/SKILL.md")

    # Add refs based on tags
    for tag in tags:
        if tag in TAG_TO_REFS:
            for ref in TAG_TO_REFS[tag]:
                refs.add(ref)

    # Add refs based on scenario class
    if scenario_class and scenario_class in SCENARIO_TO_REFS:
        for ref in SCENARIO_TO_REFS[scenario_class]:
            refs.add(ref)

    return sorted(list(refs))

def main():
    if len(sys.argv) < 2:
        print("Usage: python gen_reference_subset.py <tag1> <tag2> ... [--scenario <class>]")
        sys.exit(1)

    tags = []
    scenario_class = None
    
    args = sys.argv[1:]
    if "--scenario" in args:
        idx = args.index("--scenario")
        if idx + 1 < len(args):
            scenario_class = args[idx + 1]
            tags = args[:idx] + args[idx+2:]
        else:
            tags = args[:idx]
    else:
        tags = args

    subset = generate_subset(tags, scenario_class)
    print(json.dumps(subset, indent=2))

if __name__ == "__main__":
    main()
