import pytest

from sigalike import shifted_sigmoid_similarity
from sigalike.similarity import _preprocess


def test_identical_strings():
    assert shifted_sigmoid_similarity("hello world", "hello world") == 1.0
    assert shifted_sigmoid_similarity("HELLO WORLD", "hello world") == 1.0
    assert shifted_sigmoid_similarity("...", "...", preprocess=False) == 1.0


def test_matching_sets():
    assert shifted_sigmoid_similarity("hello world", "hello hello world") == 1.0
    assert shifted_sigmoid_similarity("hello world", "hello HELLO world") == 1.0


def test_completely_different_strings():
    assert shifted_sigmoid_similarity("apple", "orange") == 0.0


def test_wrong_input_type():
    with pytest.raises(TypeError):
        shifted_sigmoid_similarity(2, "hello")


def test_empty_strings():
    with pytest.raises(ValueError):
        shifted_sigmoid_similarity("", "")

    with pytest.raises(ValueError):
        shifted_sigmoid_similarity("???", "hello", preprocess=True)

    with pytest.raises(ValueError):
        shifted_sigmoid_similarity("    ", "hello", preprocess=True)


def test_different_strings_with_matches():
    assert round(shifted_sigmoid_similarity("hello", "hello world", shift=4), 2) == 0.95
    assert round(shifted_sigmoid_similarity("hello", "HELLO world", shift=4), 2) == 0.95
    assert round(shifted_sigmoid_similarity("hello", "HELLO, world!", shift=4), 2) == 0.95
    assert round(shifted_sigmoid_similarity("hello", "hello beautiful world", shift=4), 2) == 0.88
    assert round(shifted_sigmoid_similarity("hello", "hello beautiful WORLD", shift=4), 2) == 0.88
    assert round(shifted_sigmoid_similarity("hello", "Hello! Beautiful world...", shift=4), 2) == 0.88


def test_preprocessing():
    assert _preprocess(""",<.>/?;:'"[{]}\\|=+-_)(*&^%$#@!`~)]""") == ""
    assert _preprocess(" ") == ""
    assert _preprocess("\t") == ""
    assert _preprocess("\n") == ""
    assert _preprocess("hello-world") == "hello world"
    assert _preprocess("HELLO") == "hello"
    assert _preprocess("HELLO!!!") == "hello"
