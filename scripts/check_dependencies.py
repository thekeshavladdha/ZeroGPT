"""Check that requirements.txt stays within the project's dependency policy."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"numpy", "pandas"}
REQUIREMENT = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\s*(?:==|!=|~=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+)?$"
)


def disallowed_requirements(path: Path) -> list[str]:
    errors = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        requirement = line.split("#", 1)[0].strip()
        if not requirement:
            continue
        match = REQUIREMENT.fullmatch(requirement)
        if match is None:
            errors.append(f"line {line_number}: unsupported requirement syntax: {requirement}")
            continue
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if name not in ALLOWED:
            errors.append(f"line {line_number}: {name} is not an allowed dependency")
    return errors


def main() -> int:
    errors = disallowed_requirements(ROOT / "requirements.txt")
    if errors:
        print("Dependency policy check failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Dependency policy OK: only NumPy and optional pandas are allowed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
