"""Audit Step 2 retrieval using real Markdown section IDs.

Distinguishes:
- Real knowledge-base section IDs, such as REF-2 and RET-4.
- Other evidence identifiers, such as product SKUs.

This prevents non-section identifiers from inflating missing-section counts.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

KB_DIR = ROOT / "Capstone_Project_Dataset_FTTT_2026" / "kb"
INPUT_FILE = ROOT / "outputs" / "step1_step2_comparison.csv"
OUTPUT_FILE = ROOT / "outputs" / "step2_retrieval_audit.csv"

HEADING_PATTERN = re.compile(
    r"^\s*#{1,6}\s+\[([^\]]+)\]",
    flags=re.MULTILINE,
)


def split_ids(value: str) -> set[str]:
    """Split semicolon/comma-separated evidence identifiers."""
    if not value:
        return set()

    values = re.split(r"[;,|]+", value)

    return {
        item.strip().strip("[]").upper()
        for item in values
        if item.strip().strip("[]")
    }


def load_real_section_ids() -> set[str]:
    """Read actual section IDs from Markdown headings."""
    if not KB_DIR.is_dir():
        raise FileNotFoundError(
            f"Knowledge-base folder not found: {KB_DIR}"
        )

    markdown_files = sorted(KB_DIR.glob("*.md"))

    if not markdown_files:
        raise FileNotFoundError(
            f"No Markdown files found in: {KB_DIR}"
        )

    section_ids: set[str] = set()

    for path in markdown_files:
        text = path.read_text(encoding="utf-8-sig")

        for match in HEADING_PATTERN.finditer(text):
            section_id = match.group(1).strip().upper()

            if section_id:
                section_ids.add(section_id)

    if not section_ids:
        raise RuntimeError(
            "No real section IDs were found in the Markdown documents."
        )

    print(f"Markdown documents scanned: {len(markdown_files)}")
    print(f"Unique real section IDs found: {len(section_ids)}")

    return section_ids


def main() -> None:
    """Create a corrected retrieval audit."""
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Comparison CSV not found: {INPUT_FILE}"
        )

    real_section_ids = load_real_section_ids()

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(
            f"No rows found in comparison CSV: {INPUT_FILE}"
        )

    audit_rows = []
    questions_with_gold_sections = 0
    fully_retrieved = 0
    missing_section_count = 0
    questions_with_unmapped_ids = 0

    for row in rows:
        question_id = row.get("question_id", "")
        question = row.get("question", "")

        # The comparison file may contain both section IDs and non-section
        # identifiers in this column. Validate them against the real KB.
        all_gold_ids = split_ids(row.get("gold_evidence_ids", ""))

        gold_section_ids = all_gold_ids & real_section_ids
        non_section_ids = all_gold_ids - real_section_ids

        retrieved_candidates = split_ids(
            row.get("step2_retrieved_section_ids", "")
        )

        # Only IDs which exist in the Markdown knowledge base count as
        # retrieved knowledge-base sections.
        retrieved_section_ids = (
            retrieved_candidates & real_section_ids
        )

        matched_ids = gold_section_ids & retrieved_section_ids
        missing_ids = gold_section_ids - retrieved_section_ids

        if gold_section_ids:
            questions_with_gold_sections += 1

            if not missing_ids:
                fully_retrieved += 1

            missing_section_count += len(missing_ids)

            evidence_recall = (
                f"{len(matched_ids) / len(gold_section_ids) * 100:.2f}%"
            )
        else:
            evidence_recall = "N/A"

        if non_section_ids:
            questions_with_unmapped_ids += 1

        audit_rows.append({
            "question_id": question_id,
            "question": question,
            "gold_section_ids": "; ".join(sorted(gold_section_ids)),
            "retrieved_section_ids": "; ".join(
                sorted(retrieved_section_ids)
            ),
            "matched_section_ids": "; ".join(sorted(matched_ids)),
            "missing_section_ids": "; ".join(sorted(missing_ids)),
            "non_section_evidence_ids": "; ".join(
                sorted(non_section_ids)
            ),
            "gold_section_recall": evidence_recall,
            "generated_answer": row.get("step2_answer", ""),
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "question_id",
        "question",
        "gold_section_ids",
        "retrieved_section_ids",
        "matched_section_ids",
        "missing_section_ids",
        "non_section_evidence_ids",
        "gold_section_recall",
        "generated_answer",
    ]

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    missing_rows = [
        row for row in audit_rows
        if row["missing_section_ids"]
    ]

    unmapped_rows = [
        row for row in audit_rows
        if row["non_section_evidence_ids"]
    ]

    print("\n" + "=" * 60)
    print("CORRECTED STEP 2 RETRIEVAL AUDIT")
    print("=" * 60)
    print(f"Questions audited: {len(audit_rows)}")
    print(
        "Questions with real gold section IDs: "
        f"{questions_with_gold_sections}"
    )
    print(f"All gold sections retrieved: {fully_retrieved}")
    print(
        "Questions missing real sections: "
        f"{len(missing_rows)}"
    )
    print(f"Missing section occurrences: {missing_section_count}")
    print(
        "Questions with non-section evidence IDs: "
        f"{questions_with_unmapped_ids}"
    )
    print(f"Audit saved to: {OUTPUT_FILE}")

    print("\nFirst 10 questions with missing real sections:")

    for item in missing_rows[:10]:
        print(f"\n{item['question_id']}: {item['question']}")
        print(f"Expected sections: {item['gold_section_ids']}")
        print(f"Retrieved sections: {item['retrieved_section_ids']}")
        print(f"Missing sections: {item['missing_section_ids']}")

    print("\nFirst 5 questions with non-section evidence IDs:")

    for item in unmapped_rows[:5]:
        print(f"\n{item['question_id']}: {item['question']}")
        print(
            "IDs needing separate interpretation: "
            f"{item['non_section_evidence_ids']}"
        )

    print(
        "\nNote: an ID absent from Markdown headings is classified as a "
        "non-section identifier. That does not prove it is irrelevant; "
        "it may represent a product or another type of evidence."
    )


if __name__ == "__main__":
    main()