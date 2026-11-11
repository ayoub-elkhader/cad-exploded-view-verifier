"""LLM answer parsing (no network, no API key)."""
from exploded_view_verifier.llm import build_prompt, parse_vectors


def test_parse_plain_json():
    fix, err = parse_vectors('{"a": [1, 2, 3], "b": [0, 0, 1]}', ["a", "b"])
    assert err is None and fix == {"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 1.0]}


def test_parse_markdown_fence_and_filters_unknown_ids():
    raw = 'Sure:\n```json\n{"a": [1, 2, 3], "zzz": [1, 1, 1], "b": [1, 2]}\n```'
    fix, err = parse_vectors(raw, ["a", "b"])
    assert fix == {"a": [1.0, 2.0, 3.0]}


def test_parse_errors():
    assert parse_vectors("", ["a"]) == (None, "empty answer")
    assert parse_vectors("no json here", ["a"])[1] == "no JSON object found"
    assert parse_vectors('{"a": "up"}', ["a"])[1] == "no usable vector"
    assert parse_vectors("{not json}", ["a"])[1].startswith("invalid JSON")


def test_prompt_mentions_previous_failure():
    p = build_prompt({"colliding_pairs": [["a", "b"]]}, previous_conflicts=[("a", "b")])
    assert "STILL OVERLAPPING" in p and "LEFT 1 OVERLAPPING" in p
