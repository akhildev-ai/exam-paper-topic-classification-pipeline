from src.segment_questions import segment_paper
from src.utils import normalize_text_lines


def test_segment_paper_handles_sections_subquestions_and_marks() -> None:
    sample = normalize_text_lines(
        "Section A\n"
        "1. Define the limit of a function. [2]\n"
        "(a) Evaluate the derivative at x = 2. 3M\n"
        "Section B\n"
        "2) Find the area enclosed between the curve and x-axis. (5 Marks)"
    )

    questions = segment_paper(sample, 2023, "demo_paper")

    assert [question["question_number"] for question in questions] == ["1", "1(a)", "2"]
    assert questions[0]["marks"] == 2
    assert questions[1]["marks"] == 3
    assert questions[2]["section"] == "B"
    assert questions[2]["text"] == "Find the area enclosed between the curve and x-axis."
