from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.main import PipelineRunner
from src.utils import save_json, setup_logging

setup_logging("INFO")
st.set_page_config(page_title="Exam Topic Analysis Dashboard", layout="wide")
st.title("Exam Question Paper Analysis & Topic Classification")
st.caption("Upload exam papers and a topics config, then run the pipeline and inspect JSON outputs and topic trends.")


def _write_uploaded_file(uploaded_file: Any, target_dir: Path) -> Path:
    target_path = target_dir / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def _render_topic_charts(aggregate_report: dict[str, Any]) -> None:
    topic_totals = aggregate_report.get("topic_totals", {})
    if topic_totals:
        topic_frame = pd.DataFrame(
            [{"topic": topic, "count": count} for topic, count in topic_totals.items()]
        )
        st.plotly_chart(px.bar(topic_frame, x="topic", y="count", title="Questions per Topic"), use_container_width=True)

    trend_rows: list[dict[str, Any]] = []
    for topic, rows in aggregate_report.get("topic_trends", {}).items():
        for row in rows:
            trend_rows.append({"topic": topic, "year": row["year"], "count": row["count"]})
    if trend_rows:
        trend_frame = pd.DataFrame(trend_rows)
        st.plotly_chart(
            px.line(trend_frame, x="year", y="count", color="topic", markers=True, title="Topic Trends Across Years"),
            use_container_width=True,
        )


uploaded_papers = st.file_uploader(
    "Upload exam papers (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True
)
uploaded_topics = st.file_uploader("Upload topics YAML", type=["yaml", "yml"])
uploaded_validation = st.file_uploader("Upload validation dataset JSON", type=["json"])
use_llm_fallback = st.checkbox("Enable local LLM fallback", value=False)
llm_model_name = st.text_input("Local LLM model name", value="Qwen/Qwen2.5-7B-Instruct")

if st.button("Run analysis", type="primary"):
    if not uploaded_papers or not uploaded_topics:
        st.error("Upload at least one paper and a topics config before running the analysis.")
    else:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_dir = temp_dir / "papers"
            output_dir = temp_dir / "outputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            topics_path = _write_uploaded_file(uploaded_topics, temp_dir)
            validation_path = None
            if uploaded_validation is not None:
                validation_path = _write_uploaded_file(uploaded_validation, temp_dir)

            for uploaded_paper in uploaded_papers:
                _write_uploaded_file(uploaded_paper, input_dir)

            runner = PipelineRunner(
                input_dir=input_dir,
                topics_config_path=topics_path,
                output_dir=output_dir,
                validation_path=validation_path,
                use_llm_fallback=use_llm_fallback,
                llm_model_name=llm_model_name if use_llm_fallback else None,
            )
            result = runner.run()
            aggregate_report = result["aggregate_report"]
            validation_report = result["validation_report"]

            st.subheader("Summary")
            st.json(result["pipeline_summary"])

            st.subheader("Topic Charts")
            _render_topic_charts(aggregate_report)

            st.subheader("Reports")
            st.json(aggregate_report)
            if validation_report is not None:
                st.subheader("Validation")
                st.json(validation_report["metrics"])

            classified_json = output_dir / "questions_classified.json"
            aggregate_json = output_dir / "aggregate_report.json"
            st.download_button(
                label="Download classified questions",
                data=classified_json.read_text(encoding="utf-8"),
                file_name="questions_classified.json",
                mime="application/json",
            )
            st.download_button(
                label="Download aggregate report",
                data=aggregate_json.read_text(encoding="utf-8"),
                file_name="aggregate_report.json",
                mime="application/json",
            )
            if validation_report is not None:
                validation_json = output_dir / "validation_report.json"
                st.download_button(
                    label="Download validation report",
                    data=validation_json.read_text(encoding="utf-8"),
                    file_name="validation_report.json",
                    mime="application/json",
                )
