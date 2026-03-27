import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from keyword_extractor import KeywordExtractor
from question_generator import QuestionGenerator, QuizQuestion
from quiz_evaluator import QuizEvaluator


TEXT = (
    "Machine learning is a subset of AI that allows computers to learn from data. "
    "Supervised learning uses labelled data. Gradient descent minimizes loss. "
    "Clustering groups similar data points. Neural networks use layers of neurons."
)



@pytest.fixture
def extractor():
    return KeywordExtractor(max_keywords=10)

@pytest.fixture
def generator():
    return QuestionGenerator(num_questions=5)

@pytest.fixture
def evaluator():
    return QuizEvaluator()


def test_keywords_basic(extractor):
    res = extractor.extract(TEXT)
    assert isinstance(res, list)
    assert len(res) <= 10

def test_keywords_structure(extractor):
    for k in extractor.extract(TEXT):
        assert "keyword" in k and "score" in k
        assert 0 <= k["score"] <= 1

def test_keywords_edge_cases(extractor):
    assert isinstance(extractor.extract(""), list)
    assert isinstance(extractor.extract("Short text."), list)


def test_generate_output(generator):
    res = generator.generate(TEXT, topic="ML")
    assert res and res.topic == "ML"
    assert res.generated >= 1

def test_question_quality(generator):
    res = generator.generate(TEXT)
    for q in res.questions:
        assert len(q.options) == 4
        assert q.answer in q.options
        assert len(set(q.options)) == 4
        assert q.difficulty in {"easy", "medium", "hard"}

def test_question_limit(generator):
    generator.num_questions = 3
    res = generator.generate(TEXT)
    assert res.generated <= 3


def make_q(n=4):
    return [
        QuizQuestion(
            question=f"Q{i}",
            options=["a", "b", "c", "d"],
            answer="a",
            explanation="test",
            difficulty="medium"
        )
        for i in range(n)
    ]

def test_scoring(evaluator):
    q = make_q(4)

    perfect = evaluator.evaluate(q, ["a"] * 4)
    assert perfect.score == 100 and perfect.grade == "A"

    zero = evaluator.evaluate(q, ["x"] * 4)
    assert zero.score == 0 and zero.grade == "F"

    partial = evaluator.evaluate(q, ["a", "x", "a", "x"])
    assert partial.score == 50

def test_validation(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate(make_q(4), ["a"])

def test_difficulty_stats(evaluator):
    res = evaluator.evaluate(make_q(2), ["a", "a"])
    assert set(res.difficulty_stats.keys()) == {"easy", "medium", "hard"}
