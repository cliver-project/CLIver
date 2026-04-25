"""Tests for CallContext — per-call mutable state container."""

from cliver.llm.call_context import CallContext


class TestCallContext:
    def test_defaults(self):
        ctx = CallContext()
        assert ctx.tool_call_count == 0
        assert ctx.generated_media == []
        assert ctx.tool_result_cache == {}

    def test_independent_instances(self):
        ctx1 = CallContext()
        ctx2 = CallContext()
        ctx1.tool_call_count = 5
        ctx1.generated_media.append("img1")
        ctx1.tool_result_cache["key"] = ["result"]
        assert ctx2.tool_call_count == 0
        assert ctx2.generated_media == []
        assert ctx2.tool_result_cache == {}
