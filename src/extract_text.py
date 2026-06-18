from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import Any

from src.utils import infer_year_from_name, normalize_text_lines

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF with PyMuPDF while preserving page boundaries."""
    text_parts: list[str] = []
    try:
        fitz = import_module("fitz")

        with fitz.open(file_path) as doc:
            for page in doc:
                text_parts.append(page.get_text("text"))
    except ModuleNotFoundError:
        LOGGER.error("PyMuPDF is required for PDF extraction. Install dependencies before processing %s", file_path)
        return ""
    except Exception as exc:  # pragma: no cover - defensive logging for runtime files
        LOGGER.exception("Failed to read PDF %s: %s", file_path, exc)
        return ""
    return "\n".join(text_parts)


def extract_text_from_txt(file_path: Path) -> str:
    """Extract text from UTF-8 TXT files, falling back to replacement for broken characters."""
    try:
        return file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - defensive logging for runtime files
        LOGGER.exception("Failed to read TXT %s: %s", file_path, exc)
        return ""


def extract_text(file_path: Path) -> str:
    """Route extraction based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".txt":
        return extract_text_from_txt(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def build_paper_record(file_path: Path) -> dict[str, Any]:
    """Create a paper metadata record for downstream segmentation."""
    text = normalize_text_lines(extract_text(file_path))
    year = infer_year_from_name(file_path.stem)
    return {
        "year": year,
        "paper_name": file_path.stem,
        "file_path": str(file_path),
        "file_type": file_path.suffix.lower().lstrip("."),
        "text": text,
    }


def extract_papers_from_directory(input_dir: Path) -> list[dict[str, Any]]:
    """Extract all supported papers from a directory recursively."""
    paper_records: list[dict[str, Any]] = []
    if not input_dir.exists():
        LOGGER.warning("Input directory does not exist: %s", input_dir)
        return paper_records

    files = sorted(
        [path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    if not files:
        LOGGER.warning("No PDF/TXT files found in %s", input_dir)

    for file_path in files:
        record = build_paper_record(file_path)
        if record["text"]:
            paper_records.append(record)
        else:
            LOGGER.warning("Skipping empty extract for %s", file_path)

    LOGGER.info("Extracted %s papers from %s", len(paper_records), input_dir)
    return paper_records
