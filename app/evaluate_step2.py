"""Evaluate the Step 2 RAG chatbot against the 52 practice questions.

Results are saved to outputs/step2_rag.csv.
Successful question IDs are skipped on subsequent runs.
Failed questions are retried automatically on the next run.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from app.rag_chatbot import respond_with_rag


ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = (
    ROOT
    / "Capstone_Project_Dataset_FTTT_2026"
    / "practice"
    / "questions.csv"
)

OUTPUT_FILE = ROOT / "outputs" / "step2_rag.csv"

OUTPUT_FIELDS = [
    "question_id",
    "question",
    "language",
    "intent",
    "expected_decision",
    "acceptable_decisions",
    "answerable",
    "gold_evidence",
    "reference_answer",
    "must_include",
    "must_not_include",
    "generated_answer",
    "retrieved_section_ids",
    "retrieved_sources",
    "provider",
    "model",
    "attempted_providers",
    "status",
    "error",
]


def read_questions() -> list[dict[str, str]]:
    """Read the practice questions CSV."""
    if not QUESTIONS_FILE.is_file():
        raise FileNotFoundError(
            f"Practice questions file not found:\n{QUESTIONS_FILE}"
        )

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        questions = list(reader)

    if not questions:
        raise ValueError(
            f"No practice questions were found in {QUESTIONS_FILE}"
        )

    if "id" not in (reader.fieldnames or []):
        raise ValueError(
            "The practice questions CSV must contain an 'id' column."
        )

    if "question" not in (reader.fieldnames or []):
        raise ValueError(
            "The practice questions CSV must contain a 'question' column."
        )

    return questions


def load_existing_results() -> dict[str, dict[str, str]]:
    """Load previous results, keeping the latest row for each question ID."""
    if not OUTPUT_FILE.is_file():
        return {}

    results: dict[str, dict[str, str]] = {}

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            question_id = (row.get("question_id") or "").strip()

            if question_id:
                results[question_id] = {
                    field: row.get(field, "") or ""
                    for field in OUTPUT_FIELDS
                }

    return results


def save_results(results: dict[str, dict[str, str]]) -> None:
    """Atomically save results to avoid leaving a partially written CSV."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    with temporary_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()

        for question_id in sorted(results):
            row = results[question_id]
            writer.writerow({
                field: row.get(field, "")
                for field in OUTPUT_FIELDS
            })

    os.replace(temporary_file, OUTPUT_FILE)


def create_base_row(question: dict[str, str]) -> dict[str, str]:
    """Preserve the original labels and reference information."""
    return {
        "question_id": (question.get("id") or "").strip(),
        "question": (question.get("question") or "").strip(),
        "language": question.get("language", ""),
        "intent": question.get("intent", ""),
        "expected_decision": question.get("expected_decision", ""),
        "acceptable_decisions": question.get("acceptable_decisions", ""),
        "answerable": question.get("answerable", ""),
        "gold_evidence": question.get("gold_evidence", ""),
        "reference_answer": question.get("reference_answer", ""),
        "must_include": question.get("must_include", ""),
        "must_not_include": question.get("must_not_include", ""),
        "generated_answer": "",
        "retrieved_section_ids": "",
        "retrieved_sources": "",
        "provider": "",
        "model": "",
        "attempted_providers": "",
        "status": "pending",
        "error": "",
    }


def evaluate_question(
    question: dict[str, str],
) -> dict[str, str]:
    """Generate a RAG response and capture its retrieval metadata."""
    row = create_base_row(question)
    question_id = row["question_id"]
    prompt = row["question"]

    if not question_id:
        row["status"] = "failed"
        row["error"] = "Question has no ID."
        return row

    if not prompt:
        row["status"] = "failed"
        row["error"] = "Question text is empty."
        return row

    try:
        result: dict[str, Any] = respond_with_rag(
            question=prompt,
            top_k=8,
        )

        answer = str(result.get("answer") or "").strip()

        if not answer:
            raise RuntimeError("The chatbot returned an empty answer.")

        sources = result.get("sources") or []

        section_ids: list[str] = []
        source_details: list[str] = []

        for source in sources:
            section_id = str(source.get("section_id") or "").strip()
            filename = str(
                source.get("source_file") or "unknown"
            ).strip()

            if section_id and section_id not in section_ids:
                section_ids.append(section_id)

            label = f"[{section_id}]" if section_id else "(no section ID)"
            detail = f"{filename}:{label}"

            if detail not in source_details:
                source_details.append(detail)

        attempted = result.get("attempted_providers") or []

        row.update({
            "generated_answer": answer,
            "retrieved_section_ids": "; ".join(section_ids),
            "retrieved_sources": "; ".join(source_details),
            "provider": str(result.get("provider") or ""),
            "model": str(result.get("model") or ""),
            "attempted_providers": "; ".join(
                str(provider) for provider in attempted
            ),
            "status": "success",
            "error": "",
        })

    except Exception as exc:
        # Store the failure so the run can continue.
        # The next execution retries this question automatically.
        row["status"] = "failed"
        row["error"] = (
            f"{type(exc).__name__}: {str(exc)}"
        )[:1000]

    return row


def main() -> None:
    """Run the evaluation with resumable progress."""
    print("ZentroMart Step 2 RAG Evaluation")
    print("=" * 60)

    questions = read_questions()
    results = load_existing_results()

    # Skip only successful results. Previous failed rows will be retried.
    completed_ids = {
        question_id
        for question_id, row in results.items()
        if row.get("status") == "success"
    }

    pending_questions = [
        question
        for question in questions
        if (question.get("id") or "").strip() not in completed_ids
    ]

    print(f"Practice questions found: {len(questions)}")
    print(f"Already completed successfully: {len(completed_ids)}")
    print(f"Questions to process or retry: {len(pending_questions)}")
    print(f"Output file: {OUTPUT_FILE}")
    print()

    if not pending_questions:
        print("All questions already have successful responses.")
    else:
        for number, question in enumerate(pending_questions, start=1):
            question_id = (question.get("id") or "").strip()
            prompt = (question.get("question") or "").strip()

            print(
                f"[{number}/{len(pending_questions)}] "
                f"{question_id}: {prompt[:100]}"
            )

            row = evaluate_question(question)
            results[row["question_id"]] = row

            # Save after every question so progress survives interruptions.
            save_results(results)

            if row["status"] == "success":
                print(
                    f"  SUCCESS | Provider: {row['provider']} | "
                    f"Sections: {row['retrieved_section_ids'] or 'None'}"
                )
            else:
                print(f"  FAILED | {row['error']}")

            print()

    # Report results against the current dataset, not unrelated old rows.
    dataset_ids = {
        (question.get("id") or "").strip()
        for question in questions
        if (question.get("id") or "").strip()
    }

    current_results = [
        results[question_id]
        for question_id in dataset_ids
        if question_id in results
    ]

    successful = sum(
        row.get("status") == "success"
        for row in current_results
    )

    failed = sum(
        row.get("status") == "failed"
        for row in current_results
    )

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Questions in dataset: {len(questions)}")
    print(f"Successful responses: {successful}")
    print(f"Failed responses: {failed}")
    print(f"Output saved to: {OUTPUT_FILE}")

    if failed:
        print(
            "\nSome questions failed. Run the same command again to "
            "retry failed questions; successful responses will be skipped."
        )

    print(
        "\nNote: successful generation does not mean the answer is "
        "factually correct. Answer-quality scoring is a separate step."
    )


if __name__ == "__main__":
    main()