"""Tests for Claude/Gemini/OpenCode response parsing and artifact extraction."""


def test_claude_parse_response_success():
    from cliver.agents.claude_agent import ClaudeAgent
    from cliver.config import AgentConfig

    agent = ClaudeAgent(name="test", config=AgentConfig(type="claude"))
    raw = {
        "result": "Here is the fix.",
        "is_error": False,
        "model": "claude-sonnet-4-20250514",
        "usage": {"input": 50, "output": 100},
    }
    result = agent._parse_response(raw)
    assert result.text == "Here is the fix."
    assert result.status == "completed"
    assert result.model == "claude-sonnet-4-20250514"
    assert result.token_usage == {"input": 50, "output": 100}
    assert result.error is None


def test_claude_parse_response_error():
    from cliver.agents.claude_agent import ClaudeAgent
    from cliver.config import AgentConfig

    agent = ClaudeAgent(name="test", config=AgentConfig(type="claude"))
    raw = {"result": "Rate limit exceeded", "is_error": True}
    result = agent._parse_response(raw)
    assert result.status == "error"
    assert result.error == "Rate limit exceeded"


def test_claude_extract_artifacts():
    from cliver.agents.claude_agent import ClaudeAgent
    from cliver.config import AgentConfig

    agent = ClaudeAgent(name="test", config=AgentConfig(type="claude"))
    raw = {
        "result": "Done.",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"name": "Write", "input": {"file_path": "/tmp/app.py"}},
                    {"name": "Edit", "input": {"file_path": "/tmp/utils.py"}},
                    {"name": "Read", "input": {"file_path": "/tmp/readme.md"}},
                ],
            },
        ],
    }
    artifacts = agent._extract_artifacts(raw)
    assert len(artifacts) == 2
    assert artifacts[0].path == "/tmp/app.py"
    assert artifacts[1].path == "/tmp/utils.py"


def test_claude_extract_artifacts_no_messages():
    from cliver.agents.claude_agent import ClaudeAgent
    from cliver.config import AgentConfig

    agent = ClaudeAgent(name="test", config=AgentConfig(type="claude"))
    assert agent._extract_artifacts({"result": "no tools"}) == []


def test_claude_defaults():
    from cliver.agents.claude_agent import ClaudeAgent

    assert ClaudeAgent.DEFAULT_COMMAND == "claude"
    assert ClaudeAgent.DEFAULT_ARGS == ["-p"]
    assert ClaudeAgent.DEFAULT_OUTPUT_FORMAT == ["--output-format", "json"]


def test_gemini_parse_response_success():
    from cliver.agents.gemini_agent import GeminiAgent
    from cliver.config import AgentConfig

    agent = GeminiAgent(name="test", config=AgentConfig(type="gemini"))
    raw = {
        "response": "Analysis complete.",
        "stats": {"token_usage": {"input": 30, "output": 60}},
    }
    result = agent._parse_response(raw)
    assert result.text == "Analysis complete."
    assert result.status == "completed"
    assert result.token_usage == {"input": 30, "output": 60}


def test_gemini_parse_response_error():
    from cliver.agents.gemini_agent import GeminiAgent
    from cliver.config import AgentConfig

    agent = GeminiAgent(name="test", config=AgentConfig(type="gemini"))
    raw = {"response": "", "error": {"message": "Invalid API key", "code": 401}}
    result = agent._parse_response(raw)
    assert result.status == "error"
    assert result.error == "Invalid API key"


def test_gemini_extract_artifacts():
    from cliver.agents.gemini_agent import GeminiAgent
    from cliver.config import AgentConfig

    agent = GeminiAgent(name="test", config=AgentConfig(type="gemini"))
    raw = {
        "response": "Done.",
        "stats": {
            "file_modifications": [
                {"path": "/tmp/output.py"},
                {"path": "/tmp/image.png"},
            ]
        },
    }
    artifacts = agent._extract_artifacts(raw)
    assert len(artifacts) == 2
    assert artifacts[0].path == "/tmp/output.py"
    assert artifacts[1].path == "/tmp/image.png"
    assert artifacts[1].media_type == "image/png"


def test_gemini_defaults():
    from cliver.agents.gemini_agent import GeminiAgent

    assert GeminiAgent.DEFAULT_COMMAND == "gemini"
    assert GeminiAgent.DEFAULT_ARGS == ["-p"]


def test_opencode_parse_response_success():
    from cliver.agents.opencode_agent import OpenCodeAgent
    from cliver.config import AgentConfig

    agent = OpenCodeAgent(name="test", config=AgentConfig(type="opencode"))
    raw = {"response": "Here is the result."}
    result = agent._parse_response(raw)
    assert result.text == "Here is the result."
    assert result.status == "completed"


def test_opencode_parse_response_with_result_key():
    from cliver.agents.opencode_agent import OpenCodeAgent
    from cliver.config import AgentConfig

    agent = OpenCodeAgent(name="test", config=AgentConfig(type="opencode"))
    raw = {"result": "Fallback key."}
    result = agent._parse_response(raw)
    assert result.text == "Fallback key."


def test_opencode_parse_response_error():
    from cliver.agents.opencode_agent import OpenCodeAgent
    from cliver.config import AgentConfig

    agent = OpenCodeAgent(name="test", config=AgentConfig(type="opencode"))
    raw = {"response": "", "error": {"message": "Model not found"}}
    result = agent._parse_response(raw)
    assert result.status == "error"
    assert result.error == "Model not found"


def test_opencode_defaults():
    from cliver.agents.opencode_agent import OpenCodeAgent

    assert OpenCodeAgent.DEFAULT_COMMAND == "opencode"
    assert OpenCodeAgent.DEFAULT_ARGS == ["-p"]
    assert OpenCodeAgent.DEFAULT_OUTPUT_FORMAT == ["-f", "json"]
