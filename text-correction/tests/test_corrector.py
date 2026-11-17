import pytest
from corrector import SpellCorrector, evaluate


@pytest.fixture(scope="module")
def sc():
    return SpellCorrector()


def test_spelling(sc):
    assert sc.correct("I recieve mail definately.")["corrected"] == "I receive mail definitely."


def test_case_preserved(sc):
    assert sc.correct("Speling matters")["corrected"].startswith("Spelling")


def test_context_confusable(sc):
    assert "They're going" in sc.correct("Their going home.")["corrected"]


def test_grammar_rules(sc):
    out = sc.correct("i bought a apple the the other day")["corrected"]
    assert out == "I bought an apple the other day"


def test_change_offsets(sc):
    text = "I beleive you"
    c = sc.correct(text)["changes"][0]
    assert text[c["start"]:c["end"]] == c["original"] == "beleive"


def test_accuracy_floor(sc):
    assert evaluate(sc)[0] >= 0.85
