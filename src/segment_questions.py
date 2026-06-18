from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.utils import clean_whitespace, extract_marks_and_clean_text

LOGGER = logging.getLogger(__name__)

SECTION_PATTERN = re.compile(r"^(?:section|part)\s*[-:]?\s*([A-Za-z0-9]+)", flags=re.IGNORECASE)
NUMBER_WITH_SUB_PATTERN = re.compile(
    r"^(?:q(?:uestion)?\s*)?(?P<num>\d{1,3})\s*[\(\[]\s*(?P<sub>[A-Za-z])\s*[\)\]]\s*[\.:\-]?\s*(?P<text>.*)$"
)
NUMBER_PATTERN = re.compile(r"^(?:q(?:uestion)?\s*)?(?P<num>\d{1,3})\s*[\).:\-]?\s+(?P<text>.*)$")
SUB_PATTERN = re.compile(r"^[\(\[]?(?P<sub>[A-Za-z])[\)\].:\-]\s+(?P<text>.*)$")


class QuestionBuilder:
    """Incrementally build a question while parsing paper lines."""

    def __init__(self, year: Optional[int], paper_name: str, section: str) -> None:
        self.year = year
        self.paper_name = paper_name
        self.section = section
        self.question_number: Optional[str] = None
        self.text_parts: list[str] = []

    def reset(self, section: str, question_number: str, initial_text: str) -> None:
        self.section = section
        self.question_number = question_number
        self.text_parts = [initial_text] if initial_text else []

    def append_text(self, line: str) -> None:
        if line:
            self.text_parts.append(line)

    def to_payload(self) -> Optional[dict[str, Any]]:
        if not self.question_number:
            return None

        raw_text = clean_whitespace(" ".join(self.text_parts))
        if not raw_text:
            return None

        marks, cleaned_text = extract_marks_and_clean_text(raw_text)
        return {
            "year": self.year,
            "paper": self.paper_name,
            "paper_name": self.paper_name,
            "section": self.section,
            "question_number": self.question_number,
            "marks": marks,
            "text": cleaned_text,
        }


def segment_paper(text: str, year: Optional[int], paper_name: str) -> list[dict[str, Any]]:
    """Split full paper text into individual questions and preserve metadata."""
    questions: list[dict[str, Any]] = []
    current_section = "General"
    current_main_number: Optional[str] = None
    builder = QuestionBuilder(year=year, paper_name=paper_name, section=current_section)

    def flush_current() -> None:
        payload = builder.to_payload()
        if payload:
            questions.append(payload)

    for raw_line in text.splitlines():
        line = clean_whitespace(raw_line)
        if not line:
            continue

        section_match = SECTION_PATTERN.match(line)
        if section_match:
            flush_current()
            current_section = section_match.group(1).upper()
            builder = QuestionBuilder(year=year, paper_name=paper_name, section=current_section)
            current_main_number = None
            continue

        number_sub_match = NUMBER_WITH_SUB_PATTERN.match(line)
        if number_sub_match:
            flush_current()
            current_main_number = number_sub_match.group("num")
            sub = number_sub_match.group("sub").lower()
            text_part = number_sub_match.group("text")
            builder = QuestionBuilder(year=year, paper_name=paper_name, section=current_section)
            builder.reset(current_section, f"{current_main_number}({sub})", text_part)
            continue

        number_match = NUMBER_PATTERN.match(line)
        if number_match:
            flush_current()
            current_main_number = number_match.group("num")
            text_part = number_match.group("text")
            builder = QuestionBuilder(year=year, paper_name=paper_name, section=current_section)
            builder.reset(current_section, current_main_number, text_part)
            continue

        sub_match = SUB_PATTERN.match(line)
        if sub_match and current_main_number:
            flush_current()
            sub = sub_match.group("sub").lower()
            text_part = sub_match.group("text")
            builder = QuestionBuilder(year=year, paper_name=paper_name, section=current_section)
            builder.reset(current_section, f"{current_main_number}({sub})", text_part)
            continue

        if builder.question_number:
            builder.append_text(line)

    flush_current()
    LOGGER.info("Segmented %s questions from paper %s", len(questions), paper_name)
    return questions


def segment_all_papers(paper_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Segment all paper records into individual question payloads."""
    segmented: list[dict[str, Any]] = []
    for record in paper_records:
        questions = segment_paper(
            text=record.get("text", ""),
            year=record.get("year"),
            paper_name=record.get("paper_name", "unknown_paper"),
        )
        segmented.extend(questions)

    LOGGER.info("Total segmented questions: %s", len(segmented))
    return segmented
