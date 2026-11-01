"""Live: MiniMax TTS -> ASR round trip through the unified media pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from cliver_core import AgentCore, ModelBindings, presets
from cliver_core.media import MediaType
from cliver_core.tests.live.support import LIVE_OFF_REASON, live_enabled, skip_unless_keys

pytestmark = pytest.mark.skipif(not live_enabled(), reason=LIVE_OFF_REASON)


async def test_minimax_tts_asr_roundtrip(tmp_path: Path) -> None:
    skip_unless_keys("MINIMAX_API_KEY")
    bindings = ModelBindings(
        llm=presets.minimax_anthropic(temperature=0),
        tts=presets.minimax_tts(),
        asr=presets.minimax_asr(),
    )
    agent = AgentCore(bindings, run_timeout=180)

    try:
        speech = await agent.tts("The quick brown fox jumps over the lazy dog.", output_dir=str(tmp_path))
        assert speech.media
        audio = speech.media[0]
        assert audio.type == MediaType.AUDIO
        assert audio.saved_path
        audio_path = Path(audio.saved_path)
        assert audio_path.exists()
        assert audio_path.stat().st_size > 0

        transcript = await agent.asr(str(audio_path), language="en")
        text = (transcript.message.content or "").lower()
        assert "fox" in text or "quick" in text, f"unexpected transcript: {text[:200]!r}"
    finally:
        await agent.aclose()
