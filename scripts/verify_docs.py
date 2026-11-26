#!/usr/bin/env python3
"""Check documentation anchors against the working tree.

verified_commit in docs/knowledge-graph.json is the source commit the line
numbers were taken from. A docs-only descendant does not move those lines.
This script checks the files on disk, not that HEAD equals verified_commit.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [
    ROOT / "AGENTS.md",
    ROOT / "docs" / "engineering_standards.md",
    ROOT / "docs" / "knowledge-graph.md",
    ROOT / "docs" / "knowledge-graph.json",
    *sorted((ROOT / "docs" / "wiki").glob("*.md")),
]
CITE = re.compile(
    r"(?<![\w./-])((?:Sources|Tests|Examples|docs|Vendor|\.github|scripts)/[A-Za-z0-9_./-]+\.(?:swift|md|yml|txt|py)):(\d+)-(\d+)"
)
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def check_range(path_text: str, start: int, end: int, errors: list[str], where: str) -> None:
    if start < 1 or end < start:
        errors.append(f"{where}: bad range {path_text}:{start}-{end}")
        return
    path = ROOT / path_text
    if not path.is_file():
        errors.append(f"{where}: missing file {path_text}")
        return
    count = line_count(path)
    if end > count:
        errors.append(f"{where}: {path_text}:{start}-{end} exceeds {count} lines")


def walk_json(value, errors: list[str], where: str) -> None:
    if isinstance(value, dict):
        if {"file_path", "line_start", "line_end"} <= value.keys():
            start = value["line_start"]
            end = value["line_end"]
            if start is not None or end is not None:
                check_range(str(value["file_path"]), int(start), int(end), errors, where)
        for child in value.values():
            walk_json(child, errors, where)
    elif isinstance(value, list):
        for child in value:
            walk_json(child, errors, where)


def check_links(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for match in LINK.finditer(text):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.is_file():
            errors.append(f"{path.relative_to(ROOT)}: broken link {match.group(1)}")


def main() -> int:
    errors: list[str] = []
    for path in DOC_FILES:
        if not path.is_file():
            errors.append(f"missing doc {path.relative_to(ROOT)}")
            continue
        if path.suffix == ".json":
            walk_json(json.loads(path.read_text(encoding="utf-8")), errors, str(path.relative_to(ROOT)))
            continue
        text = path.read_text(encoding="utf-8")
        for match in CITE.finditer(text):
            check_range(match.group(1), int(match.group(2)), int(match.group(3)), errors, str(path.relative_to(ROOT)))
        check_links(path, errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"verified {len(DOC_FILES)} docs against {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
