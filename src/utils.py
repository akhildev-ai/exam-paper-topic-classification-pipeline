from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

LOGGER = logging.getLogger(__name__)

MARK_PATTERNS = [
    re.compile(r"\[(?P<marks>\d{1,2})\]", flags=re.IGNORECASE),
    re.compile(r"\((?P<marks>\d{1,2})\s*(?:marks?|m)\)", flags=re.IGNORECASE),
    re.compile(r"\b(?P<marks>\d{1,2})\s*(?:marks?|m)\b", flags=re.IGNORECASE),
]


def setup_logging(level: str = "INFO") -> None:
    """Configure application logging with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim text."""
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_text_lines(text: str) -> str:
    """Normalize each line while preserving line boundaries for downstream parsers."""
    normalized_lines = [clean_whitespace(line) for line in (text or "").splitlines()]
    return "\n".join(line for line in normalized_lines if line)


def infer_year_from_name(name: str) -> Optional[int]:
    """Extract a plausible year from a filename or text token."""
    match = re.search(r"(?<!\d)(19\d{2}|20\d{2}|21\d{2})(?!\d)", name)
    if not match:
        return None
    return int(match.group(1))


def extract_marks_and_clean_text(text: str) -> tuple[Optional[int], str]:
    """Find marks allocation patterns and remove the selected token from text."""
    normalized = clean_whitespace(text)
    if not normalized:
        return None, ""

    candidates: list[tuple[int, int, int]] = []
    for pattern in MARK_PATTERNS:
        for match in pattern.finditer(normalized):
            marks = int(match.group("marks"))
            candidates.append((match.start(), match.end(), marks))

    if not candidates:
        return None, normalized

    # Prefer marks token closest to the question tail to avoid grabbing math constants.
    start, end, marks = max(candidates, key=lambda item: item[0])
    cleaned = clean_whitespace(normalized[:start] + " " + normalized[end:])
    cleaned = re.sub(r"\(\s*\)|\[\s*\]", " ", cleaned)
    cleaned = clean_whitespace(cleaned)
    return marks, cleaned


def load_topics_config(path: Path) -> dict[str, Any]:
    """Load and validate topics YAML structure."""
    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}

    if "topics" not in config or not isinstance(config["topics"], list):
        raise ValueError("topics.yaml must contain a top-level 'topics' list")

    for item in config["topics"]:
        if "name" not in item or "keywords" not in item:
            raise ValueError("Each topic must include 'name' and 'keywords'")
        if not isinstance(item["keywords"], list):
            raise ValueError("Topic keywords must be a list")
        if "aliases" in item and not isinstance(item["aliases"], list):
            raise ValueError("Topic aliases must be a list when provided")
        if "description" in item and not isinstance(item["description"], str):
            raise ValueError("Topic description must be a string when provided")

    return config


def save_json(path: Path, data: Any) -> None:
    """Persist JSON payload with deterministic formatting."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Read a JSON file and return the deserialized object."""
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)
