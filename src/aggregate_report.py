from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Sequence

LOGGER = logging.getLogger(__name__)


def build_aggregate_report(classified_questions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate topic counts, year-wise distributions, and trend views."""
    topic_totals: Counter[str] = Counter()
    yearly_topic_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unclassified_questions: list[dict[str, Any]] = []
    multi_topic_questions: list[dict[str, Any]] = []

    for question in classified_questions:
        year = str(question.get("year") or "unknown")
        topics = question.get("topics", [])
        if not topics:
            unclassified_questions.append(question)
            continue

        if len(topics) > 1:
            multi_topic_questions.append(question)

        for topic in topics:
            topic_name = topic["name"]
            topic_totals[topic_name] += 1
            yearly_topic_counts[year][topic_name] += 1

    topic_trends: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for year in sorted(yearly_topic_counts.keys()):
        year_counts = yearly_topic_counts[year]
        for topic_name, count in sorted(year_counts.items()):
            topic_trends[topic_name].append({"year": int(year) if year.isdigit() else year, "count": count})

    report = {
        "summary": {
            "total_questions": len(classified_questions),
            "classified_questions": len(classified_questions) - len(unclassified_questions),
            "unclassified_questions": len(unclassified_questions),
            "multi_topic_questions": len(multi_topic_questions),
            "topics_covered": len(topic_totals),
        },
        "topic_totals": dict(sorted(topic_totals.items())),
        "yearly_topic_distribution": {
            year: dict(sorted(counts.items())) for year, counts in sorted(yearly_topic_counts.items())
        },
        "topic_trends": dict(sorted(topic_trends.items())),
        "unclassified_questions": unclassified_questions,
        "multi_topic_questions": multi_topic_questions,
    }
    LOGGER.info("Built aggregate report for %s classified questions", len(classified_questions))
    return report
