from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

from src.aggregate_report import build_aggregate_report
from src.classify_topics import HybridTopicClassifier
from src.extract_text import extract_papers_from_directory
from src.segment_questions import segment_all_papers
from src.utils import ensure_directory, load_json, load_topics_config, save_json, setup_logging
from src.validate import validate_predictions

LOGGER = logging.getLogger(__name__)


DEFAULT_OUTPUTS = {
    "papers": "papers_extracted.json",
    "questions": "questions_segmented.json",
    "classified": "questions_classified.json",
    "aggregate": "aggregate_report.json",
    "validation": "validation_report.json",
}


class PipelineRunner:
    """Orchestrates extraction, segmentation, classification, aggregation, and validation."""

    def __init__(
        self,
        input_dir: Path,
        topics_config_path: Path,
        output_dir: Path,
        validation_path: Optional[Path] = None,
        semantic_model_name: str = "all-MiniLM-L6-v2",
        use_llm_fallback: bool = False,
        llm_model_name: Optional[str] = None,
    ) -> None:
        self.input_dir = input_dir
        self.topics_config_path = topics_config_path
        self.output_dir = ensure_directory(output_dir)
        self.validation_path = validation_path
        self.topics_config = load_topics_config(topics_config_path)
        self.classifier = HybridTopicClassifier(
            topics_config=self.topics_config,
            semantic_model_name=semantic_model_name,
            use_llm_fallback=use_llm_fallback,
            llm_model_name=llm_model_name,
        )

    def run(self) -> dict[str, Any]:
        paper_records = extract_papers_from_directory(self.input_dir)
        segmented_questions = segment_all_papers(paper_records)
        classified_questions = self.classifier.classify_records(segmented_questions)
        aggregate_report = build_aggregate_report(classified_questions)

        artifacts = {
            "papers": paper_records,
            "questions": segmented_questions,
            "classified": classified_questions,
            "aggregate": aggregate_report,
        }

        for key, filename in DEFAULT_OUTPUTS.items():
            if key == "validation":
                continue
            save_json(self.output_dir / filename, artifacts[key])

        validation_report: Optional[dict[str, Any]] = None
        if self.validation_path and self.validation_path.exists():
            validation_records = load_json(self.validation_path)
            validation_report = validate_predictions(self.classifier, validation_records)
            save_json(self.output_dir / DEFAULT_OUTPUTS["validation"], validation_report)

        pipeline_summary = {
            "artifacts": {key: str(self.output_dir / filename) for key, filename in DEFAULT_OUTPUTS.items()},
            "summary": aggregate_report["summary"],
            "validation_available": validation_report is not None,
        }
        LOGGER.info("Pipeline completed with %s classified questions", len(classified_questions))
        return {
            "paper_records": paper_records,
            "segmented_questions": segmented_questions,
            "classified_questions": classified_questions,
            "aggregate_report": aggregate_report,
            "validation_report": validation_report,
            "pipeline_summary": pipeline_summary,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exam paper topic classification pipeline")
    parser.add_argument("--input-dir", type=Path, default=Path("data"), help="Directory containing PDF/TXT exam papers")
    parser.add_argument("--topics", type=Path, default=Path("configs/topics.yaml"), help="Path to topics.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for JSON outputs")
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/validation_questions.json"),
        help="Path to labeled validation dataset",
    )
    parser.add_argument(
        "--semantic-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model for semantic matching",
    )
    parser.add_argument(
        "--use-llm-fallback",
        action="store_true",
        help="Enable local open-source LLM fallback for low-confidence questions",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional local LLM model name, for example Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    runner = PipelineRunner(
        input_dir=args.input_dir,
        topics_config_path=args.topics,
        output_dir=args.output_dir,
        validation_path=args.validation,
        semantic_model_name=args.semantic_model,
        use_llm_fallback=args.use_llm_fallback,
        llm_model_name=args.llm_model,
    )
    result = runner.run()
    LOGGER.info("Pipeline summary: %s", result["pipeline_summary"])


if __name__ == "__main__":
    main()
