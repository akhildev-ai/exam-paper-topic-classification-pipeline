# Exam Paper Topic Classification Pipeline

## Problem Statement

Educational institutions accumulate large archives of exam question papers across many years. Manual review of those papers is slow, inconsistent, and hard to scale. This project automates the pipeline required to:

1. extract text from PDF and TXT papers,
2. segment papers into structured question-level records,
3. classify each question into one or more academic topics,
4. aggregate topic trends across years, and
5. validate predictions against human-labeled questions.

All primary outputs are produced as JSON so the system can be used both as a batch pipeline and as a backend for a dashboard.

## Solution Summary

The solution is a modular Python 3.10+ pipeline composed of five stages:

1. `Extraction`: Reads PDF papers with PyMuPDF and TXT papers with UTF-8 fallback handling.
2. `Segmentation`: Detects sections, numbered questions, sub-questions, and common marks patterns.
3. `Classification`: Uses a hybrid strategy combining configurable keyword rules, semantic similarity with SentenceTransformers, and an optional local LLM fallback.
4. `Aggregation`: Computes total topic counts, year-wise distributions, trend data, unclassified questions, and multi-topic questions.
5. `Validation`: Compares predictions against a labeled dataset and reports accuracy, precision, recall, and F1 score.

## Repository Structure

- `src/extract_text.py`: PDF/TXT extraction and paper record creation.
- `src/segment_questions.py`: Question parsing and metadata preservation.
- `src/classify_topics.py`: Hybrid topic classifier.
- `src/aggregate_report.py`: Aggregation and trend report generation.
- `src/validate.py`: Validation metrics and record-level evaluation.
- `src/main.py`: CLI pipeline orchestration.
- `app.py`: Streamlit dashboard for uploads, charts, and report downloads.
- `configs/topics.yaml`: Configurable topic definitions and keywords.
- `data/validation_questions.json`: 50 labeled validation questions.
- `outputs/`: Generated JSON artifacts.

## Part 1: Question Extraction and Segmentation

The extraction and segmentation stages are designed to handle typical exam formatting variation:

- main question numbering such as `1`, `2`, `3`
- sub-questions such as `(a)`, `5(b)`
- multi-section papers such as `Section A`, `Section B`, `Part C`
- marks formats such as `[4]`, `(5 Marks)`, `10M`

Each segmented record preserves the metadata required by the problem statement:

```json
{
	"year": 2023,
	"paper": "sample_exam_2023",
	"paper_name": "sample_exam_2023",
	"section": "B",
	"question_number": "3",
	"marks": 6,
	"text": "Find the area enclosed between the curve y = x^2 and the x-axis."
}
```

## Part 2: Topic Classification

Topic definitions are fully configurable through `configs/topics.yaml`, so new topics or keywords can be added without code changes.

### Classification Strategy

The classifier is hybrid by design:

1. `Keyword matching`: Fast deterministic scoring using topic names, aliases, and keywords.
2. `Semantic similarity`: SentenceTransformers embeddings using `all-MiniLM-L6-v2`.
3. `Optional LLM fallback`: Local open-source instruct model such as `Qwen/Qwen2.5-7B-Instruct` for low-confidence cases.

### Model Choice

- `SentenceTransformers all-MiniLM-L6-v2`
	Reason: lightweight, fast, open-source, and suitable for semantic retrieval/classification on commodity hardware.
- `Qwen2.5-7B-Instruct` as optional fallback
	Reason: open-source, instruction-tuned, and feasible within the stated 24GB VRAM budget when used selectively.

### Classification Output Format

```json
{
	"topics": [
		{
			"name": "Algebra",
			"confidence": 0.6,
			"keyword_score": 0.6,
			"semantic_score": 0.0,
			"matched_keywords": ["equation", "matrix"],
			"source": "hybrid"
		}
	],
	"multi_topic": false,
	"classification_status": "classified",
	"top_confidence": 0.6
}
```

The classifier avoids duplicate topic assignments by deduplicating selected topics before returning results.

## Part 3: Aggregation and Trend Analysis

The reporting layer generates:

1. total questions per topic,
2. year-wise topic distribution,
3. per-topic trend data across years,
4. unclassified questions, and
5. multi-topic questions.

Example aggregate summary from the bundled sample run:

```json
{
	"summary": {
		"total_questions": 6,
		"classified_questions": 6,
		"unclassified_questions": 0,
		"multi_topic_questions": 0,
		"topics_covered": 4
	},
	"topic_totals": {
		"Algebra": 1,
		"Calculus": 3,
		"Geometry": 1,
		"Probability": 1
	}
}
```

## Part 4: Confidence and Validation

Validation is implemented in `src/validate.py` and compares predicted topics against a 50-question labeled dataset.

### Current Validation Result

Latest run from `outputs/validation_report.json`:

- Accuracy: `0.74`
- Precision: `0.9762`
- Recall: `0.7593`
- F1 Score: `0.8542`

These numbers were produced from the local run included in this repository. In that run, semantic similarity was automatically disabled because `sentence-transformers` was not installed in the local environment, so the classifier fell back to keyword-based hybrid scoring only. Once the dependency is installed, semantic matching is enabled automatically without code changes.

## System Design Notes

### Why this design fits the constraints

- `Open-source only`: PyMuPDF, SentenceTransformers, Transformers, Streamlit, Plotly, and PyYAML are open-source.
- `<=24GB VRAM`: The default semantic model is lightweight, and the optional LLM path is only used for fallback classification.
- `Process within 30 minutes`: The pipeline is batch-oriented, uses lightweight parsing rules, and computes embeddings lazily.
- `JSON outputs`: Every major artifact is written as JSON.

### Robustness Features

- logging across all stages,
- defensive error handling for file extraction and model loading,
- type hints in all core modules,
- unit tests for segmentation, classification, and validation,
- Streamlit UI for interactive use.

## Dashboard

The Streamlit app supports:

1. uploading PDF and TXT papers,
2. uploading a topics config,
3. optionally uploading a validation dataset,
4. running the analysis pipeline,
5. showing topic and trend charts, and
6. downloading JSON reports.

Run it with:

```powershell
streamlit run app.py
```

## Setup and Execution

### Install

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the CLI pipeline

```powershell
py -3 -m src.main --input-dir data --topics configs/topics.yaml --output-dir outputs --validation data/validation_questions.json
```

### Optional LLM fallback

```powershell
py -3 -m src.main --input-dir data --topics configs/topics.yaml --output-dir outputs --validation data/validation_questions.json --use-llm-fallback --llm-model Qwen/Qwen2.5-7B-Instruct
```

### Run tests

```powershell
py -3 -m pytest tests -q
```

## Output Artifacts

The pipeline writes these JSON artifacts:

- `outputs/papers_extracted.json`
- `outputs/questions_segmented.json`
- `outputs/questions_classified.json`
- `outputs/aggregate_report.json`
- `outputs/validation_report.json`

## Evaluation Rubric Coverage

### Segmentation accuracy

Handled with regex-based parsing for sections, numbered questions, sub-parts, and marks extraction.

### Classification

Implemented as a configurable hybrid classifier with confidence scores and multi-topic support.

### Aggregation

Implemented through report generation for totals, yearly distributions, trends, unclassified items, and multi-topic items.

### System design

Modular architecture, JSON-first outputs, optional dashboard, and configurable topic definitions.

### Code quality

Type hints, logging, tests, error handling, and separation of responsibilities by module.

## Limitations and Future Improvements

1. Semantic matching depends on installing `sentence-transformers` locally.
2. The optional LLM fallback is implemented but should be enabled only when a suitable local model is available.
3. Segmentation currently relies on common exam patterns and can be extended further for institution-specific formatting.
4. More labeled validation data would improve calibration and threshold tuning.

## Submission Deliverables Checklist

1. `Working code`: included.
2. `Model choice`: documented above.
3. `Sample output JSON`: included in this README and in `outputs/`.
4. `Write-up`: included in this README.
