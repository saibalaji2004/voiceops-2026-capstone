"""Gnani speech-to-text and text-to-speech wrapper for Step 4."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.config import GNANI_API_KEY
from app.step3_agent import Step3Agent


def voice_turn(
    audio_path: str | Path,
    agent: Step3Agent,
    *,
    language_code: str = "en-IN",
    tts_language: str | None = None,
    voice: str = "Kaveri",
    output_file: str | Path = "outputs/voice_reply.wav",
) -> dict[str, Any]:
    """Transcribe audio, process the message, and synthesize a spoken reply."""

    source = Path(audio_path).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(
            f"Audio input file was not found: {source}"
        )

    if not GNANI_API_KEY:
        raise RuntimeError(
            "GNANI_API_KEY is missing. Add it to .env. "
            "Never place the API key in source code."
        )

    try:
        from gnani.stt import GnaniSTTClient
        from gnani.tts import AudioConfig, GnaniTTSClient
    except ImportError as exc:
        raise RuntimeError(
            "Gnani SDK is not installed. Run: "
            "python -m pip install gnani-vachana"
        ) from exc

    # Both clients use the same key.
    stt = GnaniSTTClient(api_key=GNANI_API_KEY)
    tts = GnaniTTSClient(api_key=GNANI_API_KEY)

    stt_result = stt.transcribe(
        str(source),
        language_code=language_code,
    )

    transcript = str(
        stt_result.get("transcript") or ""
    ).strip()

    if transcript:
        agent_result = agent.respond(transcript)
        reply = agent_result.text
    else:
        agent_result = None
        reply = "Sorry, I didn't catch that. Could you say it again?"

    destination = Path(output_file).expanduser()

    if not destination.is_absolute():
        destination = Path.cwd() / destination

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    locale = tts_language or language_code

    audio = tts.synthesize(
        reply,
        voice=voice,
        model="timbre-v2.5",
        language=locale,
        audio_config=AudioConfig(
            sample_rate=48000,
            encoding="linear_pcm",
            container="wav",
        ),
    )

    if not isinstance(audio, (bytes, bytearray)) or not audio:
        raise RuntimeError(
            "Gnani TTS returned no audio bytes."
        )

    destination.write_bytes(bytes(audio))

    return {
        "transcript": transcript,
        "reply": reply,
        "audio_file": str(destination),
        "agent_action": getattr(
            agent_result,
            "action",
            "empty_transcript",
        ),
        "provider": getattr(agent_result, "provider", None),
        "model": getattr(agent_result, "model", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one ZentroMart voice turn using Gnani STT and TTS."
        )
    )

    parser.add_argument(
        "audio_path",
        help="Path to a recorded WAV, MP3, or FLAC audio file",
    )

    parser.add_argument(
        "--language",
        default="en-IN",
        help="STT locale, e.g. en-IN, hi-IN, te-IN, or auto",
    )

    parser.add_argument(
        "--tts-language",
        default=None,
        help="TTS locale; defaults to --language",
    )

    parser.add_argument(
        "--voice",
        default="Kaveri",
        help="TTS voice name",
    )

    parser.add_argument(
        "--output",
        default="outputs/voice_reply.wav",
        help="Output audio file path",
    )

    args = parser.parse_args()

    agent = Step3Agent(top_k=8)

    result = voice_turn(
        args.audio_path,
        agent,
        language_code=args.language,
        tts_language=args.tts_language,
        voice=args.voice,
        output_file=args.output,
    )

    print(
        f"Transcript: {result['transcript'] or '[No speech detected]'}"
    )
    print(f"Agent reply: {result['reply']}")
    print(f"Audio saved to: {result['audio_file']}")
    print(f"Agent action: {result['agent_action']}")


if __name__ == "__main__":
    main()