# tests/test_ref_resolver.py
from cliver.workflow.ref_resolver import build_auto_context, resolve_refs


class TestResolveRefs:
    def test_input_ref(self):
        state = {"inputs": {"topic": "AI"}, "steps": {}}
        result = resolve_refs("Research ${inputs.topic}", state)
        assert result == "Research AI"

    def test_step_ref(self):
        state = {"inputs": {}, "steps": {"research": {"result": "found stuff"}}}
        result = resolve_refs("Summary: ${research.result}", state)
        assert result == "Summary: found stuff"

    def test_nested_ref(self):
        state = {"inputs": {}, "steps": {"s1": {"data": {"name": "hello"}}}}
        result = resolve_refs("Name: ${s1.data.name}", state)
        assert result == "Name: hello"

    def test_array_index_ref(self):
        state = {"inputs": {}, "steps": {"s1": {"files": [{"path": "img.png"}, {"path": "img2.png"}]}}}
        result = resolve_refs("File: ${s1.files[0].path}", state)
        assert result == "File: img.png"

    def test_missing_ref_returns_empty(self):
        state = {"inputs": {}, "steps": {}}
        result = resolve_refs("Value: ${missing.key}", state)
        assert result == "Value: "

    def test_no_refs_unchanged(self):
        state = {"inputs": {}, "steps": {}}
        result = resolve_refs("No refs here", state)
        assert result == "No refs here"

    def test_multiple_refs(self):
        state = {"inputs": {"a": "X"}, "steps": {"s1": {"b": "Y"}}}
        result = resolve_refs("${inputs.a} and ${s1.b}", state)
        assert result == "X and Y"


class TestBuildAutoContext:
    def test_single_dependency(self):
        state = {"inputs": {}, "steps": {"research": {"result": "data here"}}}
        ctx = build_auto_context(["research"], state)
        assert "[Output from step 'research']:" in ctx
        assert '"result": "data here"' in ctx

    def test_multiple_dependencies(self):
        state = {"inputs": {}, "steps": {"s1": {"a": 1}, "s2": {"b": 2}}}
        ctx = build_auto_context(["s1", "s2"], state)
        assert "[Output from step 's1']:" in ctx
        assert "[Output from step 's2']:" in ctx

    def test_no_dependencies(self):
        state = {"inputs": {}, "steps": {}}
        ctx = build_auto_context([], state)
        assert ctx == ""

    def test_missing_dependency_skipped(self):
        state = {"inputs": {}, "steps": {}}
        ctx = build_auto_context(["missing"], state)
        assert ctx == ""
