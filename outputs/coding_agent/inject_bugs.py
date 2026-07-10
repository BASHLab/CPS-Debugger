#!/usr/bin/env python3
"""Create buggy variants of the controller source code.

Usage:
    python inject_bugs.py [--verify-only]

Creates one directory per bug under outputs/coding_agent/bugs/<bug_id>/,
each containing a full copy of the source tree with exactly one modification.
"""

import json
import shutil
import sys
from pathlib import Path

from bug_specs import BUGS

ROOT = Path(__file__).resolve().parent.parent.parent  # CPS-Debugger root
ORIGINAL_SRC = ROOT / "outputs" / "phase2" / "src" / "src"
BUG_DIR = ROOT / "outputs" / "coding_agent" / "bugs"


def verify_bugs():
    """Check that every bug spec's 'original' string exists exactly once."""
    ok = True
    for bug in BUGS:
        src_file = ORIGINAL_SRC / bug["file"]
        if not src_file.exists():
            print(f"  FAIL {bug['id']}: file {bug['file']} not found")
            ok = False
            continue
        content = src_file.read_text()
        count = content.count(bug["original"])
        if count == 0:
            print(f"  FAIL {bug['id']}: original string not found in {bug['file']}")
            ok = False
        elif count > 1:
            print(f"  WARN {bug['id']}: original string found {count} times "
                  f"(will replace first occurrence only)")
        else:
            # Also verify the buggy string does NOT already appear
            if bug["buggy"] in content:
                print(f"  WARN {bug['id']}: buggy string already present in {bug['file']}")
            else:
                print(f"  OK   {bug['id']}: verified in {bug['file']}")
    return ok


def create_buggy_variant(bug):
    """Copy original source tree and apply the single-line bug."""
    bug_dir = BUG_DIR / bug["id"]
    if bug_dir.exists():
        shutil.rmtree(bug_dir)
    shutil.copytree(ORIGINAL_SRC, bug_dir)

    target_file = bug_dir / bug["file"]
    content = target_file.read_text()
    count = content.count(bug["original"])
    if count == 0:
        print(f"  ERROR: '{bug['original'][:60]}...' not found in {bug['file']}")
        return False

    modified = content.replace(bug["original"], bug["buggy"], 1)
    target_file.write_text(modified)

    # Save metadata (without ground-truth answer for the agent)
    meta = {
        "id": bug["id"],
        "file": bug["file"],
        "description": bug["description"],
        "category": bug["category"],
        "affected_function": bug["affected_function"],
        "affected_edges": bug["affected_edges"],
        "difficulty": bug["difficulty"],
        "diff": f"- {bug['original']}\n+ {bug['buggy']}",
    }
    (bug_dir / "bug_metadata.json").write_text(json.dumps(meta, indent=2))
    return True


def main():
    verify_only = "--verify-only" in sys.argv

    print("=== Verifying bug specifications against source ===\n")
    all_ok = verify_bugs()

    if verify_only:
        sys.exit(0 if all_ok else 1)

    if not all_ok:
        print("\nSome bugs failed verification. Fix them before creating variants.")
        sys.exit(1)

    print(f"\n=== Creating buggy variants in {BUG_DIR} ===\n")
    BUG_DIR.mkdir(parents=True, exist_ok=True)

    created = 0
    for bug in BUGS:
        if create_buggy_variant(bug):
            print(f"  Created: {bug['id']}")
            created += 1
        else:
            print(f"  FAILED: {bug['id']}")

    print(f"\nDone. {created}/{len(BUGS)} variants created.")


if __name__ == "__main__":
    main()
