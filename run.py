"""Batch runner for the VoiceOps 2026 capstone.

Processes audio files using:
Gnani STT -> Step3Agent -> Gnani TTS

For each input file, saves:
- A text transcript and agent reply
- A WAV audio reply
- A JSON file containing retrieved source evidence and metadata

By default, each audio file receives a fresh agent session so that
independent evaluation questions do not contaminate one another.
Use --shared-session when the files are ordered turns in one conversation.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.step3_agent import Step3Agent
from app.voice import voice_turn


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".webm",
}


def json_safe(value: Any) -> Any:
    """Convert nested objects into JSON-compatible data."""
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, default=str)
        )
    except (TypeError, ValueError):
        return str(value)


def find_audio_files(
    input_dir: Path,
    recursive: bool = False,
) -> list[Path]:
    """Find supported audio files, sorted for reproducible processing."""
    pattern = input_dir.rglob("*") if recursive else input_dir.iterdir()

    files = [
        path
        for path in pattern
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files, key=lambda path: str(path).lower())


def unique_output_stem(
    audio_file: Path,
    input_dir: Path,
    recursive: bool,
    used_stems: set[str],
) -> str:
    """Avoid overwriting outputs for files with duplicate base names."""
    if recursive:
        relative = audio_file.relative_to(input_dir)
        base = str(relative.with_suffix("")).replace("\\", "__").replace("/", "__")
    else:
        base = audio_file.stem

    if base not in used_stems:
        used_stems.add(base)
        return base

    candidate = f"{base}__{audio_file.suffix.lower().lstrip('.')}"
    counter = 2

    while candidate in used_stems:
        candidate = (
            f"{base}__{audio_file.suffix.lower().lstrip('.')}__{counter}"
        )
        counter += 1

    used_stems.add(candidate)
    return candidate


def extract_section_labels(sources: Any) -> list[str]:
    """Extract section/source labels without inventing identifiers."""
    if not isinstance(sources, list):
        return []

    labels: list[str] = []
    keys = (
        "section_id",
        "section",
        "source_id",
        "source",
        "document_id",
        "filename",
        "title",
        "id",
    )

    for source in sources:
        label = None

        if isinstance(source, str):
            label = source.strip()
        elif isinstance(source, dict):
            for key in keys:
                value = source.get(key)
                if isinstance(value, (str, int)) and str(value).strip():
                    label = str(value).strip()
                    break

        if label and label not in labels:
            labels.append(label)

    return labels


def save_text_result(
    destination: Path,
    input_file: Path,
    result: dict[str, Any],
    sources_file: Path,
) -> None:
    """Save the transcript, reply and processing metadata as readable text."""
    labels = extract_section_labels(result.get("sources"))

    content = [
        f"Input file: {input_file.name}",
        f"Transcript: {result.get('transcript', '')}",
        "",
        "Agent reply:",
        str(result.get("reply", "")),
        "",
        f"Agent action: {result.get('agent_action') or 'unknown'}",
        f"Provider: {result.get('provider') or 'not reported'}",
        f"Model: {result.get('model') or 'not reported'}",
        f"Retrieved source labels: {', '.join(labels) if labels else 'None reported'}",
        f"Full source evidence: {sources_file.name}",
    ]

    destination.write_text(
        "\n".join(content) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Process an audio folder using Gnani STT, the VoiceOps agent, "
            "and Gnani TTS."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing customer audio files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which transcripts, replies and evidence are saved.",
    )
    parser.add_argument(
        "--language",
        default="en-IN",
        help="Gnani STT language code. Default: en-IN.",
    )
    parser.add_argument(
        "--tts-language",
        default=None,
        help="TTS language code. Defaults to the STT language.",
    )
    parser.add_argument(
        "--voice",
        default="Kaveri",
        help="Gnani TTS voice. Default: Kaveri.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also discover audio files in nested input directories.",
    )
    parser.add_argument(
        "--shared-session",
        action="store_true",
        help=(
            "Reuse one agent across files in sorted order. Use this only "
            "when the files are consecutive turns from the same conversation."
        ),
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        print(
            f"ERROR: Input directory does not exist: {input_dir}",
            file=sys.stderr,
        )
        return 2

    if input_dir == output_dir or input_dir in output_dir.parents:
        print(
            "ERROR: The output directory must not be inside the input "
            "directory. This avoids processing generated outputs as inputs.",
            file=sys.stderr,
        )
        return 2

    audio_files = find_audio_files(input_dir, args.recursive)

    if not audio_files:
        print(
            f"No supported audio files found in: {input_dir}",
            file=sys.stderr,
        )
        print(
            "Supported extensions: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Audio files found: {len(audio_files)}")
    print(
        "Session mode: "
        + ("shared conversation" if args.shared_session else "isolated per file")
    )
    print()

    shared_agent = Step3Agent() if args.shared_session else None
    used_stems: set[str] = set()
    manifest: list[dict[str, Any]] = []

    for index, audio_file in enumerate(audio_files, start=1):
        stem = unique_output_stem(
            audio_file,
            input_dir,
            args.recursive,
            used_stems,
        )

        text_file = output_dir / f"{stem}.txt"
        audio_output = output_dir / f"{stem}.wav"
        sources_file = output_dir / f"{stem}.sources.json"

        print(f"[{index}/{len(audio_files)}] Processing {audio_file.name}")

        agent = shared_agent if args.shared_session else Step3Agent()

        try:
            result = voice_turn(
                audio_file,
                agent,
                language_code=args.language,
                tts_language=args.tts_language,
                voice=args.voice,
                output_file=audio_output,
            )

            safe_sources = json_safe(result.get("sources") or [])

            evidence_record = {
                "input_file": str(audio_file),
                "transcript": result.get("transcript", ""),
                "reply": result.get("reply", ""),
                "agent_action": result.get("agent_action"),
                "provider": result.get("provider"),
                "model": result.get("model"),
                "sources": safe_sources,
                "tool_result": json_safe(result.get("tool_result")),
            }

            sources_file.write_text(
                json.dumps(
                    evidence_record,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            save_text_result(
                text_file,
                audio_file,
                result,
                sources_file,
            )

            entry = {
                "input_file": str(audio_file),
                "status": "success",
                "transcript_file": str(text_file),
                "audio_reply_file": str(audio_output),
                "sources_file": str(sources_file),
                "agent_action": result.get("agent_action"),
                "provider": result.get("provider"),
                "model": result.get("model"),
            }

            manifest.append(entry)

            print(f"    Transcript/reply: {text_file.name}")
            print(f"    Audio reply:      {audio_output.name}")
            print(f"    Source evidence:  {sources_file.name}")
            print(f"    Agent action:     {result.get('agent_action')}")
            print("    Status:           SUCCESS")

        except Exception as exc:
            error_message = (
                f"Input file: {audio_file}\n"
                f"Error type: {type(exc).__name__}\n"
                f"Error: {exc}\n"
            )

            # Continue processing remaining files when one input fails.
            text_file.write_text(error_message, encoding="utf-8")

            manifest.append(
                {
                    "input_file": str(audio_file),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "error_file": str(text_file),
                }
            )

            print(f"    Status: FAILED ({type(exc).__name__}: {exc})")

    manifest_path = output_dir / "batch_manifest.json"

    manifest_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_directory": str(input_dir),
                "output_directory": str(output_dir),
                "language": args.language,
                "tts_language": args.tts_language or args.language,
                "voice": args.voice,
                "shared_session": args.shared_session,
                "total_files": len(audio_files),
                "succeeded": sum(
                    item["status"] == "success" for item in manifest
                ),
                "failed": sum(
                    item["status"] == "failed" for item in manifest
                ),
                "results": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    succeeded = sum(item["status"] == "success" for item in manifest)
    failed = len(manifest) - succeeded

    print("\nBatch complete.")
    print(f"Succeeded: {succeeded}")
    print(f"Failed:    {failed}")
    print(f"Manifest:  {manifest_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())