#!/usr/bin/env python3
"""Check documentation anchors against the working tree.

Citations are checked by hashing the cited lines. verified_commit is recorded
for humans and is not required to be an ancestor of HEAD, so a squash or
rebase does not fail this script when the file text is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "docs" / "knowledge-graph.json"
DOC_FILES = [
    ROOT / "AGENTS.md",
    ROOT / "docs" / "engineering_standards.md",
    ROOT / "docs" / "knowledge-graph.md",
    GRAPH,
    *sorted((ROOT / "docs" / "wiki").glob("*.md")),
]
CITE = re.compile(
    r"(?<![\w./-])((?:Sources|Tests|Examples|docs|Vendor|\.github|scripts)/[A-Za-z0-9_./-]+\.(?:swift|md|yml|txt|py)):(\d+)-(\d+)"
)
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def cited_text(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1 : end]) + "\n"


def content_hash(path: Path, start: int, end: int) -> str:
    digest = hashlib.sha256(cited_text(path, start, end).encode("utf-8")).hexdigest()
    return digest


def check_range(path_text: str, start: int, end: int, errors: list[str], where: str) -> str | None:
    if start < 1 or end < start:
        errors.append(f"{where}: bad range {path_text}:{start}-{end}")
        return None
    path = ROOT / path_text
    if not path.is_file():
        errors.append(f"{where}: missing file {path_text}")
        return None
    count = line_count(path)
    if end > count:
        errors.append(f"{where}: {path_text}:{start}-{end} exceeds {count} lines")
        return None
    return content_hash(path, start, end)


def collect_citations() -> dict[str, tuple[str, int, int]]:
    found: dict[str, tuple[str, int, int]] = {}

    def add(path_text: str, start: int, end: int) -> None:
        found[f"{path_text}:{start}-{end}"] = (path_text, start, end)

    for path in DOC_FILES:
        if not path.is_file() or path.suffix == ".json":
            continue
        for match in CITE.finditer(path.read_text(encoding="utf-8")):
            add(match.group(1), int(match.group(2)), int(match.group(3)))
    if GRAPH.is_file():
        walk_ranges(json.loads(GRAPH.read_text(encoding="utf-8")), add)
    return found


def walk_ranges(value, add) -> None:
    if isinstance(value, dict):
        if {"file_path", "line_start", "line_end"} <= value.keys():
            start = value["line_start"]
            end = value["line_end"]
            file_path = value["file_path"]
            if start is not None and end is not None and isinstance(file_path, str) and not file_path.startswith("http"):
                add(file_path, int(start), int(end))
        for key, child in value.items():
            if key == "content_hashes":
                continue
            walk_ranges(child, add)
    elif isinstance(value, list):
        for child in value:
            walk_ranges(child, add)


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


def write_hashes() -> int:
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}
    errors: list[str] = []
    for key, (path_text, start, end) in sorted(collect_citations().items()):
        digest = check_range(path_text, start, end, errors, "write-hashes")
        if digest is not None:
            hashes[key] = digest
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    graph["content_hashes"] = hashes
    note = graph.setdefault("metadata", {})
    note["content_hash_note"] = (
        "content_hashes are SHA-256 of the cited lines in the working tree, "
        "joined by newlines with a trailing newline. verified_commit is not "
        "checked against HEAD and does not need to be an ancestor."
    )
    GRAPH.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(hashes)} content hashes")
    return 0


def main() -> int:
    if "--write-hashes" in sys.argv:
        return write_hashes()
    errors: list[str] = []
    for path in DOC_FILES:
        if not path.is_file():
            errors.append(f"missing doc {path.relative_to(ROOT)}")
            continue
        if path.suffix != ".json":
            check_links(path, errors)
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    recorded = graph.get("content_hashes")
    if not isinstance(recorded, dict):
        errors.append("docs/knowledge-graph.json: missing content_hashes")
        recorded = {}
    citations = collect_citations()
    for key, (path_text, start, end) in sorted(citations.items()):
        digest = check_range(path_text, start, end, errors, "cite")
        expected = recorded.get(key)
        if digest is None:
            continue
        if expected is None:
            errors.append(f"cite: no content hash for {key}")
        elif expected != digest:
            errors.append(f"cite: content hash mismatch for {key}")
    for key in recorded:
        if key not in citations:
            errors.append(f"content_hashes: stale entry {key}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"verified {len(citations)} cited ranges by content hash against {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
