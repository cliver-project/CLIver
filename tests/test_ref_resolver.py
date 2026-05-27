"""Tests for ${...} reference resolution."""

import pytest


@pytest.fixture
def variables():
    return {
        "setup": {"outputs": {"domain": "AI", "agent": "cliver", "count": 10}},
        "search": {
            "outputs": {
                "text": "Found papers",
                "data": [{"title": "Paper A", "citations": 100}, {"title": "Paper B", "citations": 20}],
            }
        },
        "empty": {"outputs": {}},
    }


def test_simple_ref(variables):
    from cliver.lab.ref_resolver import resolve_refs

    result = resolve_refs("Research on ${setup.outputs.domain}", variables)
    assert result == "Research on AI"


def test_multiple_refs(variables):
    from cliver.lab.ref_resolver import resolve_refs

    result = resolve_refs("${setup.outputs.domain} with ${setup.outputs.agent}", variables)
    assert result == "AI with cliver"


def test_numeric_ref(variables):
    from cliver.lab.ref_resolver import resolve_refs

    result = resolve_refs("Count: ${setup.outputs.count}", variables)
    assert result == "Count: 10"


def test_nested_array_ref(variables):
    from cliver.lab.ref_resolver import resolve_refs

    result = resolve_refs("First: ${search.outputs.data.0.title}", variables)
    assert result == "First: Paper A"


def test_no_refs():
    from cliver.lab.ref_resolver import resolve_refs

    result = resolve_refs("No references here", {})
    assert result == "No references here"


def test_ref_not_found(variables):
    from cliver.lab.ref_resolver import resolve_refs

    with pytest.raises(ValueError, match="not found"):
        resolve_refs("${nonexistent.outputs.x}", variables)


def test_ref_field_not_found(variables):
    from cliver.lab.ref_resolver import resolve_refs

    with pytest.raises(ValueError, match="not found"):
        resolve_refs("${setup.outputs.missing_field}", variables)


def test_resolve_value_returns_python_object(variables):
    from cliver.lab.ref_resolver import resolve_value

    result = resolve_value("search.outputs.data", variables)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["title"] == "Paper A"


def test_resolve_value_int(variables):
    from cliver.lab.ref_resolver import resolve_value

    result = resolve_value("setup.outputs.count", variables)
    assert result == 10
    assert isinstance(result, int)


def test_resolve_value_not_found(variables):
    from cliver.lab.ref_resolver import resolve_value

    with pytest.raises(ValueError, match="not found"):
        resolve_value("nonexistent.outputs.x", variables)


def test_empty_template():
    from cliver.lab.ref_resolver import resolve_refs

    assert resolve_refs("", {}) == ""


def test_dollar_without_braces():
    from cliver.lab.ref_resolver import resolve_refs

    result = resolve_refs("Price is $100", {})
    assert result == "Price is $100"


def test_extract_ref_paths():
    from cliver.lab.ref_resolver import extract_ref_paths

    paths = extract_ref_paths("Use ${setup.outputs.domain} and ${search.outputs.text}")
    assert paths == ["setup.outputs.domain", "search.outputs.text"]


def test_extract_ref_paths_none():
    from cliver.lab.ref_resolver import extract_ref_paths

    assert extract_ref_paths("no refs") == []
