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


from typing import Any, Collection, Dict, Mapping, Union

from sigalike.similarity import (
    BestMatch,
    _best_match_collections,
    _best_match_string_collection,
    best_match,
)


def test_best_match_with_lists():
    collection1 = ["hello", "world"]
    collection2 = ["hello", "world", "foo"]
    expected = {
        "hello": BestMatch(match="hello", score=1.0),
        "world": BestMatch(match="world", score=1.0),
    }
    assert best_match(collection1, collection2) == expected


def test_best_match_with_tuples():
    collection1 = ("hello", "world")
    collection2 = ("hello", "world", "foo")
    expected = {
        "hello": BestMatch(match="hello", score=1.0),
        "world": BestMatch(match="world", score=1.0),
    }
    assert best_match(collection1, collection2) == expected


def test_best_match_with_sets():
    collection1 = {"hello", "world"}
    collection2 = {"hello", "world", "foo"}
    expected = {
        "hello": BestMatch(match="hello", score=1.0),
        "world": BestMatch(match="world", score=1.0),
    }
    assert best_match(collection1, collection2) == expected


def test_best_match_with_numpy_arrays():
    import numpy as np

    collection1 = np.array(["hello", "world"])
    collection2 = np.array(["hello", "world", "foo"])
    expected = {
        "hello": BestMatch(match="hello", score=1.0),
        "world": BestMatch(match="world", score=1.0),
    }
    assert best_match(collection1, collection2) == expected


def test_best_match_with_string_and_collection():
    collection = ["hello", "world", "foo"]
    assert best_match("hello", collection) == BestMatch(match="hello", score=1.0)


def test_best_match_with_string_and_mapping():
    mapping = {"hello": "world", "foo": "bar"}
    assert best_match("hello", mapping) == BestMatch(match="hello", score=1.0)


def test_best_match_with_mapping_and_collection():
    mapping = {"hello": "world", "foo": "bar"}
    collection = ["hello", "world", "foo"]
    expected = {
        "hello": BestMatch(match="hello", score=1.0),
        "foo": BestMatch(match="foo", score=1.0),
    }
    assert best_match(mapping, collection) == expected


def test_best_match_with_invalid_inputs():
    with pytest.raises(TypeError):
        best_match(2, ["hello", "world"])
    with pytest.raises(TypeError):
        best_match(["hello", "world"], 2)
    with pytest.raises(TypeError):
        best_match({"hello": "world"}, 2)
    with pytest.raises(TypeError):
        best_match(2, {"hello": "world"})


def test_best_match_with_empty_collections():
    with pytest.raises(ValueError):
        best_match([], [])

    with pytest.raises(ValueError):
        best_match(["hello"], [])

    with pytest.raises(ValueError):
        best_match([], ["hello"])

    with pytest.raises(ValueError):
        best_match({}, ["hello"])

    with pytest.raises(ValueError):
        best_match(set(), ["hello"])
