"""Step 1 prompt-only baseline evaluator using the multi-LLM router."""

import argparse
import csv
import re
import time
from pathlib import Path

from app.chatbot import SYSTEM_PROMPT
from app.config import GEMINI_MODEL
from app.llm_router import (
    AllProvidersUnavailable,
    configured_providers,
    generate_reply,
)


ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = (
    ROOT
    / "Capstone_Project_Dataset_FTTT_2026"
    / "practice"
    / "questions.csv"
)

OUTPUT_FILE = ROOT / "outputs" / "step1_baseline.csv"

INPUT_FIELDS = [
    "id",
    "language",
    "question",
    "asr_text",
    "intent",
    "expected_decision",
    "acceptable_decisions",
    "answerable",
    "gold_evidence",
    "hard_negatives",
    "must_include",
    "must_not_include",
    "reference_answer",
    "reply_language",
    "entities",
    "notes",
    "needs_native_review",
]

OUTPUT_FIELDS = INPUT_FIELDS + [
    "generated_answer",
    "provider_used",
    "model_used",
    "providers_attempted",
    "missing_required",
    "forbidden_found",
    "checklist_pass",
    "request_error",
]


def normalize(text):
    text = str(text or "").casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def split_groups(value):
    return [
        group.strip()
        for group in str(value or "").split(";")
        if group.strip()
    ]


def phrase_present(answer, phrase):
    """Find phrase matches anywhere in the answer."""

    answer_words = normalize(answer).split()
    phrase_words = normalize(phrase).split()

    if not phrase_words:
        return True

    if len(phrase_words) == 1:
        target = phrase_words[0]

        if target in answer_words:
            return True

        return (
            target == "instant"
            and "instantly" in answer_words
        )

    for start, word in enumerate(answer_words):
        if word != phrase_words[0]:
            continue

        cursor = start + 1
        matched = True

        for target in phrase_words[1:]:
            found_at = None

            for index in range(
                cursor,
                min(cursor + 2, len(answer_words)),
            ):
                if answer_words[index] == target:
                    found_at = index
                    break

            if found_at is None:
                matched = False
                break

            cursor = found_at + 1

        if matched:
            return True

    return False


def missing_required_phrases(answer, specification):
    missing = []

    for group in split_groups(specification):
        alternatives = [
            item.strip()
            for item in group.split("/")
            if item.strip()
        ]

        if alternatives and not any(
            phrase_present(answer, item)
            for item in alternatives
        ):
            missing.append(group)

    return missing


def find_forbidden_phrases(answer, specification):
    found = []

    for group in split_groups(specification):
        alternatives = [
            item.strip()
            for item in group.split("/")
            if item.strip()
        ]

        for phrase in alternatives:
            if phrase_present(answer, phrase):
                found.append(phrase)

    return found


def load_successful_results():
    """Retain successful responses, including responses marked REVIEW."""

    if not OUTPUT_FILE.exists():
        return {}

    with OUTPUT_FILE.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        old_rows = list(csv.DictReader(handle))

    successful = {}

    for row in old_rows:
        question_id = (row.get("id") or "").strip()
        answer = (row.get("generated_answer") or "").strip()
        error = (row.get("request_error") or "").strip()

        if question_id and answer and not error:
            if not row.get("provider_used"):
                row["provider_used"] = "gemini (legacy direct call)"
                row["model_used"] = GEMINI_MODEL
                row["providers_attempted"] = (
                    "gemini (legacy direct call)"
                )

            successful[question_id] = row

    return successful


def rewrite_successful_results(rows):
    """Rewrite results without failed requests, preserving successes."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows.values():
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Step 1 prompt-only baseline."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retain successful responses and retry unanswered questions.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1.")

    providers = configured_providers()

    if not providers:
        raise SystemExit(
            "No LLM providers are configured. Check the project-root .env."
        )

    if not QUESTIONS_FILE.exists():
        raise SystemExit(
            f"Practice questions file not found: {QUESTIONS_FILE}"
        )

    with QUESTIONS_FILE.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        questions = list(csv.DictReader(handle))

    if args.limit is not None:
        questions = questions[:args.limit]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        successful = load_successful_results()
        rewrite_successful_results(successful)
        completed = set(successful)

    else:
        if OUTPUT_FILE.exists():
            raise SystemExit(
                f"Output already exists: {OUTPUT_FILE}\n"
                "Use --resume or move the existing file aside."
            )

        completed = set()

    pending = [
        row for row in questions
        if row["id"] not in completed
    ]

    if not pending:
        print("No pending questions in the selected range.")
        print(f"Results saved to: {OUTPUT_FILE}")
        return

    print("Configured equal-peer providers:")

    for provider in providers:
        print(
            f"  - {provider['provider']}: {provider['model']}"
        )

    file_exists = OUTPUT_FILE.exists()

    with OUTPUT_FILE.open(
        "a" if file_exists else "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        for index, row in enumerate(pending, start=1):
            question_id = row["id"]

            print(
                f"[{index}/{len(pending)}] Processing {question_id}...",
                flush=True,
            )

            try:
                response = generate_reply(
                    [
                        {
                            "role": "user",
                            "content": row["question"],
                        }
                    ],
                    SYSTEM_PROMPT,
                )

            except AllProvidersUnavailable as exc:
                print("\nAll configured providers are unavailable.")
                print(str(exc))
                print(
                    "The current question was not recorded as complete. "
                    "Use --resume after availability returns."
                )
                break

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

                print(
                    f"Unexpected error for {question_id}: {error}"
                )

                # Resume mode will remove this failed request record.
                writer.writerow(
                    {
                        **row,
                        "generated_answer": "",
                        "provider_used": "",
                        "model_used": "",
                        "providers_attempted": "",
                        "missing_required": "",
                        "forbidden_found": "",
                        "checklist_pass": "False",
                        "request_error": error,
                    }
                )

                handle.flush()
                continue

            answer = response.text

            missing = missing_required_phrases(
                answer,
                row.get("must_include", ""),
            )

            forbidden = find_forbidden_phrases(
                answer,
                row.get("must_not_include", ""),
            )

            checklist_pass = (
                bool(answer)
                and not missing
                and not forbidden
            )

            result = {
                **row,
                "generated_answer": answer,
                "provider_used": response.provider,
                "model_used": response.model,
                "providers_attempted": " ; ".join(
                    response.attempted_providers
                ),
                "missing_required": " ; ".join(missing),
                "forbidden_found": " ; ".join(forbidden),
                "checklist_pass": str(checklist_pass),
                "request_error": "",
            }

            writer.writerow(result)
            handle.flush()

            print(
                f"    Provider: {response.provider} | "
                f"Model: {response.model}\n"
                f"    Checklist: "
                f"{'PASS' if checklist_pass else 'REVIEW'}",
                flush=True,
            )

            time.sleep(2)

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(
        "Checklist results are preliminary and are not "
        "a definitive semantic accuracy score."
    )


if __name__ == "__main__":
    main()