"""Test retrieval coverage at different top-k settings.

This evaluates retrieval only. It does not call any LLM providers.
Product IDs and other non-section evidence IDs are excluded from
section-retrieval metrics.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from app.rag import search


ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "Capstone_Project_Dataset_FTTT_2026" / "kb"
INPUT_FILE = ROOT / "outputs" / "step1_step2_comparison.csv"
OUTPUT_FILE = ROOT / "outputs" / "retrieval_k_sweep.csv"

K_VALUES = [5, 8, 10, 12]

HEADING_PATTERN = re.compile(
    r"^\s*#{1,6}\s+\[([^\]]+)\]",
    flags=re.MULTILINE,
)


def split_ids(value: str) -> set[str]:
    return {
        item.strip().strip("[]").upper()
        for item in re.split(r"[;,|]+", value or "")
        if item.strip().strip("[]")
    }


def load_real_section_ids() -> set[str]:
    """Read section IDs from actual Markdown headings."""
    section_ids: set[str] = set()

    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        section_ids.update(
            match.group(1).strip().upper()
            for match in HEADING_PATTERN.finditer(text)
        )

    if not section_ids:
        raise RuntimeError(
            f"No section IDs found in knowledge base: {KB_DIR}"
        )

    return section_ids


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Comparison file not found: {INPUT_FILE}"
        )

    if not KB_DIR.is_dir():
        raise FileNotFoundError(
            f"Knowledge-base directory not found: {KB_DIR}"
        )

    real_ids = load_real_section_ids()

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    audit_rows = []
    summaries = {
        k: {
            "questions": 0,
            "fully_retrieved": 0,
            "gold_total": 0,
            "gold_found": 0,
            "retrieved_total": 0,
        }
        for k in K_VALUES
    }

    print(f"Knowledge-base section IDs: {len(real_ids)}")
    print(f"Questions loaded: {len(rows)}")
    print("Testing top-k values:", K_VALUES)
    print("\nRunning retrieval-only evaluation...\n")

    for number, row in enumerate(rows, start=1):
        question_id = (row.get("question_id") or "").strip()
        question = (row.get("question") or "").strip()

        all_gold_ids = split_ids(row.get("gold_evidence_ids", ""))
        gold_ids = all_gold_ids & real_ids

        if not question or not gold_ids:
            continue

        # Retrieve once at the largest k, then evaluate each smaller prefix.
        results = search(question, top_k=max(K_VALUES))

        ranked_sections = [
            str(item.get("section_id", "")).strip().upper()
            for item in results
        ]

        output_row = {
            "question_id": question_id,
            "question": question,
            "gold_section_ids": "; ".join(sorted(gold_ids)),
            "non_section_evidence_ids": "; ".join(
                sorted(all_gold_ids - real_ids)
            ),
        }

        for k in K_VALUES:
            # k counts retrieved chunks, not unique sections.
            retrieved_ids = {
                section_id
                for section_id in ranked_sections[:k]
                if section_id
            }

            matched = gold_ids & retrieved_ids
            missing = gold_ids - retrieved_ids

            summary = summaries[k]
            summary["questions"] += 1
            summary["gold_total"] += len(gold_ids)
            summary["gold_found"] += len(matched)
            summary["retrieved_total"] += len(retrieved_ids)

            if not missing:
                summary["fully_retrieved"] += 1

            output_row[f"top{k}_retrieved_sections"] = "; ".join(
                sorted(retrieved_ids)
            )
            output_row[f"top{k}_matched_sections"] = "; ".join(
                sorted(matched)
            )
            output_row[f"top{k}_missing_sections"] = "; ".join(
                sorted(missing)
            )

        audit_rows.append(output_row)

        if number % 10 == 0 or number == len(rows):
            print(f"Processed {number}/{len(rows)} comparison rows")

    fields = [
        "question_id",
        "question",
        "gold_section_ids",
        "non_section_evidence_ids",
    ]

    for k in K_VALUES:
        fields.extend([
            f"top{k}_retrieved_sections",
            f"top{k}_matched_sections",
            f"top{k}_missing_sections",
        ])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    print("\n" + "=" * 64)
    print("RETRIEVAL TOP-K COMPARISON")
    print("=" * 64)

    for k in K_VALUES:
        summary = summaries[k]
        questions = summary["questions"]
        gold_total = summary["gold_total"]
        gold_found = summary["gold_found"]
        retrieved_total = summary["retrieved_total"]
        fully_retrieved = summary["fully_retrieved"]

        recall = (
            100 * gold_found / gold_total
            if gold_total else 0.0
        )
        precision = (
            100 * gold_found / retrieved_total
            if retrieved_total else 0.0
        )
        full_rate = (
            100 * fully_retrieved / questions
            if questions else 0.0
        )

        print(f"\nTop {k} chunks")
        print(
            f"Section recall: {gold_found}/{gold_total} "
            f"({recall:.2f}%)"
        )
        print(
            f"Section precision: {gold_found}/{retrieved_total} "
            f"({precision:.2f}%)"
        )
        print(
            f"All expected sections found: "
            f"{fully_retrieved}/{questions} ({full_rate:.2f}%)"
        )

    print(f"\nDetailed results saved to:\n{OUTPUT_FILE}")
    print(
        "\nThese metrics measure section retrieval, not the factual "
        "accuracy of generated answers."
    )


if __name__ == "__main__":
    main()