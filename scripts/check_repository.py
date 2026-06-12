#!/usr/bin/env python3
"""Validate the paper-reading repository without third-party dependencies."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent.parent
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
LARGE_FILE_BYTES = 25 * 1024 * 1024


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def local_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
    if "xxx." in target or target == "...":
        return None
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith(("#", "mailto:")):
        return None
    path = unquote(parsed.path)
    if not path:
        return None
    return (source.parent / path).resolve()


def main() -> int:
    files = tracked_files()
    errors: list[str] = []
    warnings: list[str] = []

    for markdown in (path for path in files if path.suffix.lower() == ".md"):
        try:
            content = markdown.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 Markdown: {markdown.relative_to(ROOT)}")
            continue

        in_fence = False
        for line_number, line in enumerate(content.splitlines(), start=1):
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for match in LINK_PATTERN.finditer(line):
                target = local_target(markdown, match.group(1))
                if target is not None and not target.exists():
                    issue = (
                        f"broken link: {markdown.relative_to(ROOT)}:{line_number} "
                        f"-> {match.group(1)}"
                    )
                    if match.group(0).startswith("!"):
                        warnings.append(issue)
                    else:
                        errors.append(issue)

    section_parents = {path.parent.parent for path in files if path.parent.name == "sections"}
    for paper_dir in sorted(section_parents):
        if not (paper_dir / "README.md").is_file():
            errors.append(f"missing README.md: {paper_dir.relative_to(ROOT)}")

    for path in files:
        if path.is_file() and path.stat().st_size > LARGE_FILE_BYTES:
            size_mib = path.stat().st_size / (1024 * 1024)
            warnings.append(f"large tracked file ({size_mib:.1f} MiB): {path.relative_to(ROOT)}")

    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    print(
        f"Checked {sum(path.suffix.lower() == '.md' for path in files)} Markdown files "
        f"and {len(files)} tracked files: {len(errors)} error(s), "
        f"{len(warnings)} warning(s)."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
