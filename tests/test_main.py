import pytest

from sigalike import shifted_sigmoid_similarity


def test_identical_strings():
    assert shifted_sigmoid_similarity("hello world", "hello world", 0) == 1.0


def test_completely_different_strings():
    assert shifted_sigmoid_similarity("apple", "orange", 0) == 0.0


def test_wrong_input_type():
    with pytest.raises(TypeError):
        shifted_sigmoid_similarity(2, "hello", 0)


def test_empty_strings():
    with pytest.raises(ValueError):
        shifted_sigmoid_similarity("", "", 0)
