from typing import Sequence

from src.classify_topics import HybridTopicClassifier


class StubEmbeddingBackend:
    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self._vectors = list(vectors)
        self._index = 0

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        count = len(texts)
        start = self._index
        end = start + count
        self._index = end
        return self._vectors[start:end]


def test_hybrid_classifier_returns_unique_multi_topic_predictions() -> None:
    config = {
        "topics": [
            {"name": "Calculus", "keywords": ["derivative", "integral"]},
            {"name": "Geometry", "keywords": ["circle", "area enclosed"]},
        ]
    }
    backend = StubEmbeddingBackend(
        vectors=[
            [1.0, 0.0],
            [0.8, 0.2],
            [0.9, 0.1],
        ]
    )
    classifier = HybridTopicClassifier(
        topics_config=config,
        embedding_backend=backend,
        classification_threshold=0.2,
        multi_topic_threshold=0.2,
        multi_topic_margin=0.2,
    )

    result = classifier.classify_question("Find the derivative of the area enclosed by the circle.")

    assert result["multi_topic"] is True
    assert {topic["name"] for topic in result["topics"]} == {"Calculus", "Geometry"}
    assert len({topic["name"] for topic in result["topics"]}) == 2
