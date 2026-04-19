"""Tests for _drain_pending_input helper in AgentCore."""

from langchain_core.messages import HumanMessage, SystemMessage

from cliver.llm.llm import _drain_pending_input


def test_no_callback_no_change():
    """When on_pending_input is None, messages should remain unchanged."""
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="user message"),
    ]
    original_length = len(messages)

    _drain_pending_input(messages, on_pending_input=None)

    assert len(messages) == original_length
    assert messages[0].content == "system"
    assert messages[1].content == "user message"


def test_drain_appends_human_messages():
    """Pending input should be appended as HumanMessage."""
    messages = [SystemMessage(content="system")]
    pending_queue = ["first input", "second input", None]
    call_count = [0]

    def mock_callback():
        result = pending_queue[call_count[0]] if call_count[0] < len(pending_queue) else None
        call_count[0] += 1
        return result

    _drain_pending_input(messages, on_pending_input=mock_callback)

    assert len(messages) == 3
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "first input"
    assert isinstance(messages[2], HumanMessage)
    assert messages[2].content == "second input"


def test_drain_empty_callback():
    """Callback returning None immediately should do nothing."""
    messages = [SystemMessage(content="system")]

    def empty_callback():
        return None

    _drain_pending_input(messages, on_pending_input=empty_callback)

    assert len(messages) == 1
    assert messages[0].content == "system"


def test_drain_called_multiple_times():
    """Each drain should only take what's available at that moment."""
    messages = [SystemMessage(content="system")]
    pending_queue = ["input1", None]
    call_count = [0]

    def mock_callback():
        result = pending_queue[call_count[0]] if call_count[0] < len(pending_queue) else None
        call_count[0] += 1
        return result

    # First drain
    _drain_pending_input(messages, on_pending_input=mock_callback)
    assert len(messages) == 2
    assert messages[1].content == "input1"

    # Reset callback state with new input
    pending_queue = ["input2", None]
    call_count[0] = 0

    # Second drain
    _drain_pending_input(messages, on_pending_input=mock_callback)
    assert len(messages) == 3
    assert messages[2].content == "input2"
