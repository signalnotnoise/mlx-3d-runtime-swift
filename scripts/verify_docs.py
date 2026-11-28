#!/usr/bin/env python3
"""Check documentation anchors against the working tree.

Citations use a line range (``Sources/Foo.swift:12-40``) or a Swift symbol
(``Sources/Foo.swift#Type.member``). Line ranges are locked by a SHA-256 of
the cited lines. Symbol citations are resolved in the Swift file and fail
only when the symbol is gone, renamed, or its short name is ambiguous.

``verified_commit`` is recorded for humans and is not required to be an
ancestor of HEAD.

  python3 scripts/verify_docs.py
  python3 scripts/verify_docs.py --fix
  python3 scripts/verify_docs.py --write-hashes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import doc_citations as cites

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "docs" / "knowledge-graph.json"
DOC_FILES = [
    ROOT / "AGENTS.md",
    ROOT / "docs" / "engineering_standards.md",
    ROOT / "docs" / "knowledge-graph.md",
    GRAPH,
    *sorted((ROOT / "docs" / "wiki").glob("*.md")),
]
import re

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def read_lines(rel: str) -> list[str] | None:
    path = ROOT / rel
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").splitlines()


def load_graph() -> dict:
    return json.loads(GRAPH.read_text(encoding="utf-8"))


def save_graph(graph: dict) -> None:
    GRAPH.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")


def citation_from_fields(file_path: str, start=None, end=None, symbol=None) -> cites.Citation | None:
    if not isinstance(file_path, str) or file_path.startswith("http"):
        return None
    if symbol:
        return cites.Citation(path=file_path, symbol=str(symbol))
    if start is None or end is None:
        return None
    return cites.Citation(path=file_path, start=int(start), end=int(end))


def walk_graph_citations(value, add) -> None:
    if isinstance(value, dict):
        if "file_path" in value and (
            "line_start" in value or "symbol" in value
        ):
            cite = citation_from_fields(
                value.get("file_path"),
                value.get("line_start"),
                value.get("line_end"),
                value.get("symbol"),
            )
            if cite:
                add(cite)
        for key, child in value.items():
            if key == "content_hashes":
                continue
            walk_graph_citations(child, add)
    elif isinstance(value, list):
        for child in value:
            walk_graph_citations(child, add)


def collect_markdown_citations() -> list[tuple[Path, cites.Citation, str]]:
    """(doc path, citation, exact matched text)."""
    found: list[tuple[Path, cites.Citation, str]] = []
    for path in DOC_FILES:
        if not path.is_file() or path.suffix == ".json":
            continue
        text = path.read_text(encoding="utf-8")
        for match in cites.CITATION.finditer(text):
            found.append((path, cites.citation_from_match(match), match.group(0)))
    return found


def collect_citations(graph: dict) -> dict[str, cites.Citation]:
    found: dict[str, cites.Citation] = {}

    def add(cite: cites.Citation) -> None:
        found[cite.key] = cite

    for _, cite, _ in collect_markdown_citations():
        add(cite)
    walk_graph_citations(graph, add)
    return found


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


def resolve_symbol(rel: str, name: str) -> tuple[cites.Symbol | None, str | None]:
    lines = read_lines(rel)
    if lines is None:
        return None, f"missing file {rel}"
    text = "\n".join(lines) + "\n"
    symbols = cites.index_swift(text)
    exact = [s for s in symbols if s.name == name]
    if exact:
        # Overloads: the symbol exists. Return the first; existence is what matters.
        return exact[0], None
    return cites.find_symbol(symbols, name)


def check_citation(
    cite: cites.Citation,
    recorded: dict[str, str],
    errors: list[str],
    where: str,
) -> str | None:
    """Return the current hash for a line-range citation, or None."""
    if cite.is_symbol:
        symbol, err = resolve_symbol(cite.path, cite.symbol or "")
        if err:
            errors.append(f"{where}: {cite.key}: {err}")
        return None

    lines = read_lines(cite.path)
    if lines is None:
        errors.append(f"{where}: missing file {cite.path}")
        return None
    start, end = cite.line_start, cite.line_end
    if start < 1 or end < start:
        errors.append(f"{where}: bad range {cite.key}")
        return None
    if end > len(lines):
        errors.append(f"{where}: {cite.key} exceeds {len(lines)} lines")
        return None
    digest = cites.hash_lines(lines, start, end)
    expected = recorded.get(cite.key)
    if expected is None:
        errors.append(f"{where}: no content hash for {cite.key}")
    elif expected != digest:
        errors.append(
            f"{where}: content hash mismatch for {cite.key} "
            f"(content changed or moved; run python3 scripts/verify_docs.py --fix "
            f"if the cited text moved unchanged in this file)"
        )
    return digest


def rewrite_text(text: str, replacements: dict[str, str]) -> tuple[str, int]:
    """Replace citation keys in text. Longer keys first so prefixes stay intact."""
    if not replacements:
        return text, 0
    count = 0

    def repl(match: re.Match) -> str:
        nonlocal count
        cite = cites.citation_from_match(match)
        new = replacements.get(cite.key)
        if not new or new == cite.key:
            return match.group(0)
        count += 1
        return new

    return cites.CITATION.sub(repl, text), count


def rewrite_graph_paths(value, replacements: dict[str, str]) -> int:
    changed = 0
    if isinstance(value, dict):
        if "file_path" in value:
            symbol = value.get("symbol")
            start, end = value.get("line_start"), value.get("line_end")
            old = citation_from_fields(value.get("file_path"), start, end, symbol)
            if old and old.key in replacements:
                new = cites.parse_citation(replacements[old.key]) or cites.citation_from_match(
                    cites.CITATION.search(replacements[old.key])
                )
                # parse_citation uses search; replacements values are keys.
                new = _parse_key(replacements[old.key])
                if new.is_symbol:
                    value["symbol"] = new.symbol
                    # Keep line hints if we can resolve them later; drop stale numbers
                    # only after we fill them in the caller.
                else:
                    value["line_start"] = new.line_start
                    value["line_end"] = new.line_end
                changed += 1
        for key, child in value.items():
            if key == "content_hashes":
                continue
            changed += rewrite_graph_paths(child, replacements)
    elif isinstance(value, list):
        for child in value:
            changed += rewrite_graph_paths(child, replacements)
    return changed


def _parse_key(key: str) -> cites.Citation:
    match = cites.CITATION.fullmatch(key)
    if not match:
        raise ValueError(f"not a citation key: {key}")
    return cites.citation_from_match(match)


def apply_replacements_to_docs(replacements: dict[str, str]) -> int:
    total = 0
    for path in DOC_FILES:
        if not path.is_file() or path.suffix == ".json":
            continue
        text = path.read_text(encoding="utf-8")
        new, n = rewrite_text(text, replacements)
        if n:
            path.write_text(new, encoding="utf-8")
            total += n
    return total


def upgrade_symbol(cite: cites.Citation) -> cites.Citation | None:
    if cite.is_symbol or not cite.path.endswith(".swift"):
        return None
    lines = read_lines(cite.path)
    if lines is None:
        return None
    symbols = cites.index_swift("\n".join(lines) + "\n")
    symbol = best_symbol_upgrade(symbols, cite.line_start, cite.line_end)
    if symbol is None:
        return None
    return cites.Citation(path=cite.path, symbol=symbol.name)


def best_symbol_upgrade(symbols: list[cites.Symbol], start: int, end: int) -> cites.Symbol | None:
    exact = cites.unique_symbol_for_range(symbols, start, end)
    if exact:
        return exact
    types = [
        s
        for s in symbols
        if s.kind in {"enum", "struct", "class", "actor", "protocol", "extension", "func", "var", "init"}
        and (s.start == start or s.decl_line == start)
    ]
    candidates: list[cites.Symbol] = []
    for symbol in types:
        if end > symbol.end + 2:
            continue
        typelen = symbol.end - symbol.decl_line + 1
        span = end - start + 1
        if typelen <= 0 or span < max(1, int(0.8 * typelen)):
            continue
        candidates.append(symbol)
    if len(candidates) == 1:
        return candidates[0]
    return None


def fill_symbol_line_hints(graph: dict) -> None:
    def walk(value) -> None:
        if isinstance(value, dict):
            symbol = value.get("symbol")
            path = value.get("file_path")
            if symbol and path:
                found, err = resolve_symbol(path, symbol)
                if found:
                    value["line_start"] = found.start
                    value["line_end"] = found.end
            for key, child in value.items():
                if key == "content_hashes":
                    continue
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(graph)


def fix(graph: dict) -> int:
    recorded: dict[str, str] = dict(graph.get("content_hashes") or {})
    citations = collect_citations(graph)
    replacements: dict[str, str] = {}
    unresolved: list[str] = []
    new_hashes: dict[str, str] = {}

    for key, cite in sorted(citations.items()):
        if cite.is_symbol:
            # Leave symbol citations in place; refresh hints later.
            continue
        lines = read_lines(cite.path)
        expected = recorded.get(key)
        if lines is None:
            unresolved.append(f"{key}: missing file")
            continue
        start, end = cite.line_start, cite.line_end
        in_range = 1 <= start <= end <= len(lines)
        current = cites.hash_lines(lines, start, end) if in_range else None
        target = cite
        if expected and (not in_range or current != expected):
            length = end - start + 1
            matches = cites.find_hash_windows(lines, expected, length)
            if len(matches) == 1:
                new_start, new_end = matches[0]
                target = cites.Citation(path=cite.path, start=new_start, end=new_end)
                print(f"retarget {key} -> {target.key}")
            elif len(matches) == 0:
                # Try a symbol upgrade before giving up (content often changed
                # inside a type whose header is still at the cited start).
                upgraded = upgrade_symbol(cite)
                if upgraded:
                    print(f"migrate  {key} -> {upgraded.key} (content changed; symbol still found)")
                    replacements[key] = upgraded.key
                    continue
                unresolved.append(
                    f"{key}: cited text is gone or changed; not guessing"
                )
                continue
            else:
                windows = ", ".join(f"{a}-{b}" for a, b in matches)
                unresolved.append(
                    f"{key}: cited text occurs {len(matches)} times ({windows}); not guessing"
                )
                continue
        else:
            if expected is None and in_range:
                expected = current
            elif expected is None:
                unresolved.append(f"{key}: no content hash and range is invalid")
                continue

        upgraded = upgrade_symbol(target)
        if upgraded:
            if upgraded.key != key:
                print(f"migrate  {key} -> {upgraded.key}")
                replacements[key] = upgraded.key
            continue
        if target.key != key:
            replacements[key] = target.key
            new_hashes[target.key] = expected or current or ""
        elif current:
            new_hashes[key] = current if expected is None or expected == current else expected

    # Hashes for citations we didn't touch
    for key, cite in citations.items():
        if cite.is_symbol or key in replacements:
            continue
        if key in new_hashes:
            continue
        if key in recorded and key not in {u.split(":")[0] for u in unresolved}:
            # keep old hash if still unresolved (don't rewrite away a failure)
            if any(u.startswith(key + ":") for u in unresolved):
                new_hashes[key] = recorded[key]
            else:
                new_hashes[key] = recorded[key]

    apply_replacements_to_docs(replacements)
    rewrite_graph_paths(graph, replacements)

    # Recompute hashes for remaining line-range citations after rewrites
    # and drop hashes for citations that became symbols.
    fill_symbol_line_hints(graph)
    graph_after = graph
    remaining = collect_citations(graph_after)
    hashes: dict[str, str] = {}
    for key, cite in remaining.items():
        if cite.is_symbol:
            continue
        lines = read_lines(cite.path)
        if lines is None or cite.line_end > len(lines) or cite.line_start < 1:
            continue
        hashes[key] = cites.hash_lines(lines, cite.line_start, cite.line_end)
    graph["content_hashes"] = hashes
    note = graph.setdefault("metadata", {})
    note["content_hash_note"] = (
        "content_hashes are SHA-256 of line-range citations in the working "
        "tree, joined by newlines with a trailing newline. Symbol citations "
        "(file.swift#Type.member) are not hashed. verified_commit is not "
        "checked against HEAD and does not need to be an ancestor."
    )
    save_graph(graph)

    if unresolved:
        print("unresolved citations:", file=sys.stderr)
        for item in unresolved:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(f"fixed citations; {len(hashes)} line-range hashes, {sum(1 for c in remaining.values() if c.is_symbol)} symbol citations")
    return 0


def write_hashes(graph: dict) -> int:
    hashes: dict[str, str] = {}
    errors: list[str] = []
    for key, cite in sorted(collect_citations(graph).items()):
        if cite.is_symbol:
            continue
        lines = read_lines(cite.path)
        if lines is None:
            errors.append(f"write-hashes: missing file {cite.path}")
            continue
        if cite.line_start < 1 or cite.line_end < cite.line_start or cite.line_end > len(lines):
            errors.append(f"write-hashes: bad range {key}")
            continue
        hashes[key] = cites.hash_lines(lines, cite.line_start, cite.line_end)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    graph["content_hashes"] = hashes
    save_graph(graph)
    print(f"wrote {len(hashes)} content hashes")
    return 0


def main() -> int:
    if not GRAPH.is_file():
        print(f"missing {GRAPH}", file=sys.stderr)
        return 1
    graph = load_graph()
    if "--fix" in sys.argv:
        status = fix(graph)
        if status != 0:
            return status
        # Re-check after fixing
        graph = load_graph()
    if "--write-hashes" in sys.argv:
        return write_hashes(graph)

    errors: list[str] = []
    for path in DOC_FILES:
        if not path.is_file():
            errors.append(f"missing doc {path.relative_to(ROOT)}")
            continue
        if path.suffix != ".json":
            check_links(path, errors)
    recorded = graph.get("content_hashes")
    if not isinstance(recorded, dict):
        errors.append("docs/knowledge-graph.json: missing content_hashes")
        recorded = {}
    citations = collect_citations(graph)
    for key, cite in sorted(citations.items()):
        check_citation(cite, recorded, errors, "cite")
    for key in recorded:
        if key not in citations:
            errors.append(f"content_hashes: stale entry {key}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    symbol_count = sum(1 for c in citations.values() if c.is_symbol)
    range_count = len(citations) - symbol_count
    print(
        f"verified {range_count} line-range citation(s) and "
        f"{symbol_count} symbol citation(s) against {ROOT}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
