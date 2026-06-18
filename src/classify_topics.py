from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

from src.utils import clean_whitespace

LOGGER = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Protocol for pluggable text embedding backends."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Encode a batch of texts into embedding vectors."""


class LLMBackend(Protocol):
    """Protocol for optional local LLM fallback implementations."""

    def classify(
        self, question_text: str, topics: Sequence["TopicDefinition"], max_topics: int
    ) -> list[dict[str, Any]]:
        """Return a list of topic predictions with confidence scores."""


@dataclass(frozen=True)
class TopicDefinition:
    """Normalized topic definition loaded from the YAML configuration."""

    name: str
    keywords: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    description: str = ""

    @property
    def all_terms(self) -> tuple[str, ...]:
        terms = [self.name, *self.aliases, *self.keywords]
        unique_terms: list[str] = []
        seen: set[str] = set()
        for term in terms:
            normalized = clean_whitespace(term).lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_terms.append(term)
        return tuple(unique_terms)

    def semantic_prompt(self) -> str:
        joined_keywords = ", ".join(self.keywords)
        description = f" {self.description}" if self.description else ""
        return f"Topic: {self.name}.{description} Keywords: {joined_keywords}".strip()


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Compute cosine similarity without requiring NumPy at import time."""
    dot_product = sum(float(left) * float(right) for left, right in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(float(value) * float(value) for value in vector_a))
    norm_b = math.sqrt(sum(float(value) * float(value) for value in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class SentenceTransformerBackend:
    """Lazy SentenceTransformer wrapper used by the hybrid classifier."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        model = self._load_model()
        embeddings = model.encode(list(texts), normalize_embeddings=True)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings


class TransformersLLMBackend:
    """Optional local open-source LLM fallback for low-confidence questions."""

    def __init__(self, model_name: str, max_new_tokens: int = 128) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipeline: Any = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="text-generation",
                model=self.model_name,
                device_map="auto",
            )
        return self._pipeline

    def classify(self, question_text: str, topics: Sequence[TopicDefinition], max_topics: int) -> list[dict[str, Any]]:
        available_topics = ", ".join(topic.name for topic in topics)
        prompt = (
            "You are classifying an academic exam question. "
            "Choose up to {max_topics} topics from this list: {available_topics}. "
            "Return strict JSON with the shape "
            '{"topics": [{"name": "Topic Name", "confidence": 0.0}]}. '
            "Question: {question_text}"
        ).format(max_topics=max_topics, available_topics=available_topics, question_text=question_text)
        generator = self._load_pipeline()
        raw_output = generator(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)[0]["generated_text"]
        json_match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
        if not json_match:
            return self._fallback_parse(raw_output, topics)

        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return self._fallback_parse(raw_output, topics)

        predictions: list[dict[str, Any]] = []
        valid_topics = {topic.name.lower(): topic.name for topic in topics}
        for item in parsed.get("topics", []):
            topic_name = clean_whitespace(str(item.get("name", "")))
            canonical_name = valid_topics.get(topic_name.lower())
            if not canonical_name:
                continue
            confidence = float(item.get("confidence", 0.5))
            predictions.append(
                {
                    "name": canonical_name,
                    "confidence": round(max(0.0, min(confidence, 1.0)), 4),
                    "source": "llm_fallback",
                }
            )
        return predictions[:max_topics]

    def _fallback_parse(self, raw_output: str, topics: Sequence[TopicDefinition]) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        lowered_output = raw_output.lower()
        for topic in topics:
            if topic.name.lower() in lowered_output:
                predictions.append({"name": topic.name, "confidence": 0.55, "source": "llm_fallback"})
        return predictions


class HybridTopicClassifier:
    """Hybrid topic classifier combining keyword and semantic matching."""

    def __init__(
        self,
        topics_config: dict[str, Any],
        semantic_model_name: str = "all-MiniLM-L6-v2",
        keyword_weight: float = 0.45,
        semantic_weight: float = 0.55,
        classification_threshold: float = 0.35,
        multi_topic_threshold: float = 0.33,
        multi_topic_margin: float = 0.1,
        max_topics: int = 3,
        use_llm_fallback: bool = False,
        llm_model_name: Optional[str] = None,
        embedding_backend: Optional[EmbeddingBackend] = None,
        llm_backend: Optional[LLMBackend] = None,
    ) -> None:
        self.topics = self._build_topics(topics_config)
        self.semantic_model_name = semantic_model_name
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.classification_threshold = classification_threshold
        self.multi_topic_threshold = multi_topic_threshold
        self.multi_topic_margin = multi_topic_margin
        self.max_topics = max_topics
        self.embedding_backend = embedding_backend
        self._semantic_disabled = False
        self._topic_embeddings: Optional[Sequence[Sequence[float]]] = None
        self.use_llm_fallback = use_llm_fallback
        self.llm_backend = llm_backend or (
            TransformersLLMBackend(llm_model_name) if use_llm_fallback and llm_model_name else None
        )

    @staticmethod
    def _build_topics(topics_config: dict[str, Any]) -> list[TopicDefinition]:
        topics: list[TopicDefinition] = []
        for item in topics_config.get("topics", []):
            topics.append(
                TopicDefinition(
                    name=clean_whitespace(item["name"]),
                    keywords=tuple(clean_whitespace(keyword) for keyword in item.get("keywords", []) if keyword),
                    aliases=tuple(clean_whitespace(alias) for alias in item.get("aliases", []) if alias),
                    description=clean_whitespace(item.get("description", "")),
                )
            )
        return topics

    def _get_embedding_backend(self) -> Optional[EmbeddingBackend]:
        if self._semantic_disabled:
            return None
        if self.embedding_backend is None:
            try:
                self.embedding_backend = SentenceTransformerBackend(self.semantic_model_name)
            except Exception as exc:  # pragma: no cover - dependency/runtime guard
                LOGGER.warning("Disabling semantic matching because the backend could not be initialized: %s", exc)
                self._semantic_disabled = True
                return None
        return self.embedding_backend

    def _ensure_topic_embeddings(self) -> tuple[Optional[Sequence[Sequence[float]]], bool]:
        backend = self._get_embedding_backend()
        if backend is None:
            return None, False
        if self._topic_embeddings is None:
            try:
                self._topic_embeddings = backend.encode([topic.semantic_prompt() for topic in self.topics])
            except Exception as exc:  # pragma: no cover - dependency/runtime guard
                LOGGER.warning("Disabling semantic matching because topic embeddings failed: %s", exc)
                self._semantic_disabled = True
                self._topic_embeddings = None
                return None, False
        return self._topic_embeddings, True

    def _keyword_score(self, question_text: str, topic: TopicDefinition) -> tuple[float, list[str]]:
        lowered_text = question_text.lower()
        matched_terms: list[str] = []
        for term in topic.all_terms:
            lowered_term = term.lower()
            if not lowered_term:
                continue
            if " " in lowered_term:
                matched = lowered_term in lowered_text
            else:
                matched = re.search(rf"\b{re.escape(lowered_term)}\b", lowered_text) is not None
            if matched and term not in matched_terms:
                matched_terms.append(term)

        if not matched_terms:
            return 0.0, []

        keyword_coverage = min(1.0, len(matched_terms) / max(1.0, min(4.0, float(len(topic.all_terms)))))
        topic_name_bonus = 0.2 if topic.name.lower() in lowered_text else 0.0
        phrase_bonus = 0.15 if any(" " in term for term in matched_terms) else 0.0
        match_bonus = 0.1
        score = min(1.0, keyword_coverage + topic_name_bonus + phrase_bonus + match_bonus)
        return score, matched_terms

    def _semantic_scores(self, question_text: str) -> tuple[list[float], bool]:
        topic_embeddings, semantic_available = self._ensure_topic_embeddings()
        if not semantic_available or topic_embeddings is None:
            return [0.0] * len(self.topics), False

        backend = self._get_embedding_backend()
        if backend is None:
            return [0.0] * len(self.topics), False

        try:
            question_embedding = backend.encode([question_text])[0]
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            LOGGER.warning("Semantic encoding failed for a question; keyword-only classification will be used: %s", exc)
            self._semantic_disabled = True
            return [0.0] * len(self.topics), False

        scores = [max(0.0, cosine_similarity(question_embedding, topic_embedding)) for topic_embedding in topic_embeddings]
        return scores, True

    def classify_question(self, question_text: str) -> dict[str, Any]:
        normalized_text = clean_whitespace(question_text)
        if not normalized_text:
            return {"topics": [], "multi_topic": False, "classification_status": "unclassified", "top_confidence": 0.0}

        semantic_scores, semantic_available = self._semantic_scores(normalized_text)
        ranked_topics: list[dict[str, Any]] = []
        for index, topic in enumerate(self.topics):
            keyword_score, matched_terms = self._keyword_score(normalized_text, topic)
            semantic_score = semantic_scores[index]
            active_weight = self.keyword_weight + (self.semantic_weight if semantic_available else 0.0)
            blended_score = (
                (keyword_score * self.keyword_weight) + (semantic_score * (self.semantic_weight if semantic_available else 0.0))
            ) / active_weight
            combined_score = blended_score
            if semantic_available and keyword_score >= self.classification_threshold:
                combined_score = max(blended_score, keyword_score)
            ranked_topics.append(
                {
                    "name": topic.name,
                    "confidence": round(combined_score, 4),
                    "keyword_score": round(keyword_score, 4),
                    "semantic_score": round(semantic_score, 4),
                    "matched_keywords": matched_terms,
                    "source": "hybrid",
                }
            )

        ranked_topics.sort(key=lambda item: item["confidence"], reverse=True)
        selected_topics = self._select_topics(ranked_topics)

        if not selected_topics and self.use_llm_fallback and self.llm_backend is not None:
            try:
                selected_topics = self._merge_llm_predictions(
                    ranked_topics=ranked_topics,
                    llm_predictions=self.llm_backend.classify(normalized_text, self.topics, self.max_topics),
                )
            except Exception as exc:  # pragma: no cover - dependency/runtime guard
                LOGGER.warning("LLM fallback failed and will be ignored for this question: %s", exc)

        top_confidence = ranked_topics[0]["confidence"] if ranked_topics else 0.0
        return {
            "topics": selected_topics,
            "multi_topic": len(selected_topics) > 1,
            "classification_status": "classified" if selected_topics else "unclassified",
            "top_confidence": top_confidence,
        }

    def _select_topics(self, ranked_topics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not ranked_topics:
            return []

        top_topic = ranked_topics[0]
        if top_topic["confidence"] < self.classification_threshold:
            return []

        selected_topics = [top_topic]
        for candidate in ranked_topics[1:]:
            if len(selected_topics) >= self.max_topics:
                break
            relative_floor = top_topic["confidence"] * max(0.0, 1.0 - (self.multi_topic_margin + 0.05))
            within_margin = (top_topic["confidence"] - candidate["confidence"]) <= self.multi_topic_margin
            within_relative_range = candidate["confidence"] >= relative_floor
            above_threshold = candidate["confidence"] >= self.multi_topic_threshold
            if above_threshold and (within_margin or within_relative_range):
                selected_topics.append(candidate)

        deduplicated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for topic in selected_topics:
            if topic["name"] in seen:
                continue
            seen.add(topic["name"])
            deduplicated.append(topic)
        return deduplicated

    def _merge_llm_predictions(
        self, ranked_topics: Sequence[dict[str, Any]], llm_predictions: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        merged_by_name = {topic["name"]: dict(topic) for topic in ranked_topics[: self.max_topics]}
        for prediction in llm_predictions:
            topic_name = prediction["name"]
            existing = merged_by_name.get(topic_name, {})
            merged_by_name[topic_name] = {
                "name": topic_name,
                "confidence": round(max(float(existing.get("confidence", 0.0)), float(prediction.get("confidence", 0.0))), 4),
                "keyword_score": float(existing.get("keyword_score", 0.0)),
                "semantic_score": float(existing.get("semantic_score", 0.0)),
                "matched_keywords": existing.get("matched_keywords", []),
                "source": prediction.get("source", "llm_fallback"),
            }

        merged_topics = sorted(merged_by_name.values(), key=lambda item: item["confidence"], reverse=True)
        fallback_topics = [topic for topic in merged_topics if topic["confidence"] >= self.multi_topic_threshold]
        return fallback_topics[: self.max_topics]

    def classify_question_record(self, question_record: dict[str, Any]) -> dict[str, Any]:
        classification = self.classify_question(question_record.get("text", ""))
        return {**question_record, **classification}

    def classify_records(self, question_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.classify_question_record(question_record) for question_record in question_records]