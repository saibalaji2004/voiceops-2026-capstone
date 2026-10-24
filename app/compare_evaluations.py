"""Compare Step 1 prompt-only and Step 2 RAG evaluations.

Outputs:
    outputs/step1_step2_comparison.csv

Metrics are transparent checklist and retrieval metrics.
They do not claim to measure full semantic correctness.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = (
    ROOT
    / "Capstone_Project_Dataset_FTTT_2026"
    / "practice"
    / "questions.csv"
)

STEP1_FILE = ROOT / "outputs" / "step1_baseline.csv"
STEP2_FILE = ROOT / "outputs" / "step2_rag.csv"
OUTPUT_FILE = ROOT / "outputs" / "step1_step2_comparison.csv"
KB_DIR = ROOT / "Capstone_Project_Dataset_FTTT_2026" / "kb"
PRODUCTS_FILE = ROOT / "Capstone_Project_Dataset_FTTT_2026" / "data" / "products.csv"
OFFLINE_AUDIT_FILE = ROOT / "outputs" / "step2_offline_quality_audit.csv"

# Supported column-name alternatives make the script tolerant of
# small differences between the two existing evaluation CSVs.
ID_COLUMNS = ("question_id", "id", "questionid")
ANSWER_COLUMNS = (
    "generated_answer",
    "answer",
    "response",
    "assistant_response",
    "model_response",
    "generated_response",
    "output",
    "reply",
)
STATUS_COLUMNS = ("status", "response_status", "execution_status")
PROVIDER_COLUMNS = ("provider", "llm_provider")
MODEL_COLUMNS = ("model", "llm_model")
SECTION_COLUMNS = (
    "retrieved_section_ids",
    "section_ids",
    "evidence_section_ids",
    "retrieved_sections",
)

SECTION_ID_PATTERN = re.compile(
    r"\b[A-Z][A-Z0-9]*-\d+(?:\.\d+)?\b"
)

OUTPUT_FIELDS = [
    "question_id",
    "question",
    "expected_decision",
    "gold_evidence_ids",
    "must_include",
    "must_not_include",
    "step1_status",
    "step1_provider",
    "step1_model",
    "step1_answer_available",
    "step1_required_phrases_matched",
    "step1_required_phrases_total",
    "step1_missing_required_phrases",
    "step1_forbidden_phrase_hits",
    "step1_checklist_pass",
    "step1_answer",
    "step2_status",
    "step2_provider",
    "step2_model",
    "step2_answer_available",
    "step2_required_phrases_matched",
    "step2_required_phrases_total",
    "step2_missing_required_phrases",
    "step2_forbidden_phrase_hits",
    "step2_checklist_pass",
    "step2_answer",
    "step2_retrieved_section_ids",
    "step2_gold_evidence_recall",
    "step2_gold_evidence_precision",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV and return its rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def get_value(
    row: dict[str, Any],
    aliases: tuple[str, ...],
) -> str:
    """Return the first available field, ignoring case and whitespace."""
    normalized_row = {
        str(key).strip().lower(): value
        for key, value in row.items()
        if key is not None
    }

    for name in aliases:
        value = normalized_row.get(name.lower())

        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def index_by_question_id(
    rows: list[dict[str, str]],
    file_label: str,
) -> dict[str, dict[str, str]]:
    """Index evaluation rows by question ID and report duplicate IDs."""
    indexed: dict[str, dict[str, str]] = {}
    duplicates: set[str] = set()

    for row in rows:
        question_id = get_value(row, ID_COLUMNS)

        if not question_id:
            continue

        if question_id in indexed:
            duplicates.add(question_id)

        # Keep the latest row if duplicates exist.
        indexed[question_id] = row

    if duplicates:
        print(
            f"Warning: {file_label} contains duplicate question IDs: "
            + ", ".join(sorted(duplicates))
        )

    return indexed


def split_expected_phrases(value: str) -> list[str]:
    """Parse phrase lists separated by semicolons, pipes, or newlines."""
    if not value.strip():
        return []

    stripped = value.strip()

    # Also support JSON-like lists such as ["phrase one", "phrase two"].
    if stripped.startswith("[") and stripped.endswith("]"):
        import json

        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, list):
                return [
                    str(item).strip()
                    for item in decoded
                    if str(item).strip()
                ]
        except json.JSONDecodeError:
            pass

    phrases = re.split(r"[;|\r\n]+", stripped)

    return [
        phrase.strip().strip("\"'").strip()
        for phrase in phrases
        if phrase.strip().strip("\"'").strip()
    ]


def normalize_text(text: str) -> str:
    """Normalize case and punctuation for a simple phrase check."""
    text = text.casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(answer: str, phrase: str) -> bool:
    """Check for a whole normalized phrase rather than a substring."""
    normalized_answer = normalize_text(answer)
    normalized_phrase = normalize_text(phrase)

    if not normalized_answer or not normalized_phrase:
        return False

    return (
        f" {normalized_phrase} " in f" {normalized_answer} "
    )


def extract_section_ids(text: str) -> list[str]:
    """Extract unique section IDs such as REF-2 and KB-00.2."""
    found = SECTION_ID_PATTERN.findall(text.upper())

    return list(dict.fromkeys(found))


def row_has_answer(row: dict[str, str]) -> bool:
    """Determine whether an evaluation row has a usable response."""
    answer = get_value(row, ANSWER_COLUMNS)

    if not answer:
        return False

    status = get_value(row, STATUS_COLUMNS).casefold()

    if status in {
        "failed",
        "failure",
        "error",
        "pending",
        "missing",
        "no_response",
    }:
        return False

    return True


def evaluate_checklist(
    answer: str,
    answer_available: bool,
    must_include: list[str],
    must_not_include: list[str],
) -> dict[str, Any]:
    """Calculate the required/forbidden phrase checklist metrics."""
    matched_required = [
        phrase
        for phrase in must_include
        if answer_available and contains_phrase(answer, phrase)
    ]

    missing_required = [
        phrase
        for phrase in must_include
        if phrase not in matched_required
    ]

    forbidden_hits = [
        phrase
        for phrase in must_not_include
        if answer_available and contains_phrase(answer, phrase)
    ]

    checklist_eligible = bool(must_include or must_not_include)

    checklist_pass = (
        answer_available
        and not missing_required
        and not forbidden_hits
        if checklist_eligible
        else None
    )

    return {
        "matched_required": matched_required,
        "missing_required": missing_required,
        "forbidden_hits": forbidden_hits,
        "required_total": len(must_include),
        "required_matched": len(matched_required),
        "eligible": checklist_eligible,
        "checklist_pass": checklist_pass,
    }


def format_list(values: list[str]) -> str:
    """Format values consistently for CSV output."""
    return "; ".join(values)


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def percent(numerator: int, denominator: int) -> str:
    """Format a percentage or mark a metric not applicable."""
    if denominator == 0:
        return "N/A"

    return f"{100 * numerator / denominator:.2f}%"


def summarize_stage(
    name: str,
    rows: list[dict[str, str]],
) -> None:
    """Print response coverage and checklist metrics for one stage."""
    total = len(rows)

    answered_rows = [
        row for row in rows
        if row[f"{name.lower()}_answer_available"] == "yes"
    ]

    eligible_rows = [
        row for row in rows
        if row[f"{name.lower()}_checklist_pass"] in {"yes", "no"}
    ]

    checklist_passes = sum(
        row[f"{name.lower()}_checklist_pass"] == "yes"
        for row in eligible_rows
    )

    phrase_total = sum(
        int(row[f"{name.lower()}_required_phrases_total"])
        for row in rows
    )

    phrase_matched = sum(
        int(row[f"{name.lower()}_required_phrases_matched"])
        for row in rows
    )

    forbidden_hits = sum(
        len(split_expected_phrases(
            row[f"{name.lower()}_forbidden_phrase_hits"]
        ))
        for row in rows
    )

    print(f"\n{name}")
    print("-" * 44)
    print(
        f"Responses available: {len(answered_rows)}/{total} "
        f"({percent(len(answered_rows), total)})"
    )
    print(
        f"Required phrase coverage: {phrase_matched}/{phrase_total} "
        f"({percent(phrase_matched, phrase_total)})"
    )
    print(f"Forbidden phrase hits: {forbidden_hits}")
    print(
        f"Checklist pass rate: {checklist_passes}/{len(eligible_rows)} "
        f"({percent(checklist_passes, len(eligible_rows))})"
    )



# Offline quality-audit helpers. These use only local CSV/Markdown files and
# make no calls to hosted LLM APIs. Results are diagnostic flags, not semantic
# accuracy scores.
CITATION_PATTERN = re.compile(
    r"\[\s*([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+(?:\.\d+)?)\s*\]"
    )
SKU_PATTERN = re.compile(r"\b[A-Z]{3}-\d{4}\b", flags=re.IGNORECASE)
KB_HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s+\[([^\]]+)\]", flags=re.MULTILINE)
DASH_TRANSLATION = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})


def normalize_identifier(value: str) -> str:
    """Normalize section IDs, including typographic hyphens from model output."""
    return (value or "").strip().translate(DASH_TRANSLATION).upper()


def load_real_section_ids() -> set[str]:
    """Load genuine bracketed section IDs from the Markdown knowledge base."""
    if not KB_DIR.is_dir():
        raise FileNotFoundError(f"Knowledge-base folder not found: {KB_DIR}")

    ids: set[str] = set()
    for path in sorted(KB_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8-sig")
        ids.update(
            normalize_identifier(match.group(1))
            for match in KB_HEADING_PATTERN.finditer(content)
            if match.group(1).strip()
        )

    if not ids:
        raise RuntimeError(f"No section IDs found in Markdown files under {KB_DIR}")
    return ids


def load_product_catalogue() -> dict[str, dict[str, str]]:
    """Load product metadata for deterministic SKU consistency checks."""
    if not PRODUCTS_FILE.is_file():
        return {}

    with PRODUCTS_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            sku = normalize_identifier(row.get("sku", ""))
            if sku:
                result[sku] = {
                    str(key).strip().lower(): str(value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
        return result


def extract_ids_from_value(value: str) -> set[str]:
    """Read IDs from fields serialized as semicolon/comma separated text."""
    return {
        normalize_identifier(part.strip().strip("[]"))
        for part in re.split(r"[;,|]+", value or "")
        if part.strip().strip("[]")
    }


def answer_looks_truncated(answer: str) -> bool:
    """Flag obvious truncation signs; punctuation alone is not a failure."""
    value = (answer or "").strip()
    if not value:
        return True

    # Remove trailing citations in either [REF-2] or 【REF-2】 format.
    value = re.sub(
        r"(?:\s*[\[【][A-Za-z][A-Za-z0-9]*"
        r"(?:-[A-Za-z0-9]+)+(?:\.\d+)?[\]】])+\s*$",
        "",
        value,
    ).strip()

    if not value:
        return True

    # Very short answers are suspicious and should be reviewed manually.
    # This catches corrupted outputs such as "क्रेड" without declaring
    # every short answer incorrect.
    if len(value) < 12 or len(value.split()) < 3:
        return True

    if value.endswith(("...", "…", "—", "–", ",", ";", ":")):
        return True

    lower = value.casefold()
    dangling_endings = (
        "please note that", "please note", "because", "although",
        "which means", "and", "or", "but", "so that", "as per",
        "कृपया ध्यान दें कि", "क्योंकि", "लेकिन", "और",
        "जिसके अनुसार", "के अनुसार कि", "ध्यान दें कि",
    )

    return any(lower.endswith(ending) for ending in dangling_endings)


def decision_signal(answer: str, expected: str) -> tuple[str, str]:
    """Provide a cautious keyword-based signal for decision handling."""
    normalized = normalize_text(answer)
    decision = (expected or "").strip().casefold()

    signals = {
        "clarify": (
            "could you clarify", "please confirm", "please provide",
            "which order", "can you share", "more details", "स्पष्ट करें",
            "कृपया बताएं", "कौन सा", "कृपया जानकारी दें",
        ),
        "escalate": (
            "human support", "support colleague", "support team",
            "connect you", "escalate", "human agent", "customer care",
            "मानव सहायता", "सपोर्ट टीम", "कस्टमर केयर",
        ),
        "refuse": (
            "cannot provide", "can't provide", "cannot share", "can't share",
            "not able to share", "cannot help with that", "privacy",
            "नहीं बता सकता", "साझा नहीं कर सकता", "गोपनीयता",
        ),
    }

    matched_categories = [
        category for category, phrases in signals.items()
        if any(normalize_text(phrase) in normalized for phrase in phrases)
    ]

    if not decision:
        return "not_evaluated", "No expected decision recorded."

    # The dataset can contain multiple acceptable decisions. This is a hint
    # for manual review, never a claim that semantics were fully evaluated.
    aliases = {part.strip() for part in re.split(r"[;,|]+", decision) if part.strip()}
    category_aliases = {
        "clarify": {"clarify", "clarification"},
        "escalate": {"escalate", "escalation", "human"},
        "refuse": {"refuse", "refusal", "deny"},
    }
    target_categories = {
        category for category, accepted in category_aliases.items()
        if aliases & accepted
    }

    if target_categories:
        overlap = sorted(target_categories & set(matched_categories))
        if overlap:
            return "signal_found", "Keyword signal found for: " + ", ".join(overlap)
        return "review", "No obvious keyword signal for expected clarify/escalate/refuse decision."

    if len(normalized.split()) >= 5:
        return "substantive_response", "Answer appears substantive; semantic correctness not assessed."
    return "review", "Answer may be too short; inspect manually."


def product_consistency_flags(
    question: str,
    answer: str,
    catalogue: dict[str, dict[str, str]],
) -> tuple[str, str]:
    """Check obvious SKU and warranty contradictions against the product CSV."""
    query_skus = sorted({normalize_identifier(sku) for sku in SKU_PATTERN.findall(question)})
    answer_skus = sorted({normalize_identifier(sku) for sku in SKU_PATTERN.findall(answer)})
    if not query_skus:
        return "not_applicable", ""

    missing = [sku for sku in query_skus if sku not in catalogue]
    notes: list[str] = []
    flags: list[str] = []
    normalized_answer = normalize_text(answer)

    if missing:
        flags.append("sku_not_found_in_catalogue")
        notes.append("Question SKU(s) not found in products.csv: " + ", ".join(missing))

    for sku in query_skus:
        product = catalogue.get(sku)
        if not product:
            continue

        title = product.get("title", "")
        assured = product.get("assured", "").strip().casefold()
        fulfilled_by = product.get("fulfilled_by", "")
        warranty_value = product.get("warranty_months", "").strip()
        notes.append(
            f"{sku}: {title}; assured={assured or 'unknown'}; "
            f"fulfilled_by={fulfilled_by or 'unknown'}; "
            f"warranty_months={warranty_value or 'unknown'}"
        )

        # Only flag direct, fairly clear contradictions; don't infer from a
        # missing mention or language-specific phrasing.
        negative_assured = any(term in normalized_answer for term in (
            "not assured", "isnt assured", "is not assured",
            "does not qualify for assured", "not eligible for assured",
            "assured nahi", "assured nahin", "assured lagu nahi",
        ))
        positive_assured = (
            "zentromart assured" in normalized_answer
            or "assured hai" in normalized_answer
            or "assured true" in normalized_answer
        )

        if assured in {"true", "1", "yes"} and negative_assured:
            flags.append("possible_assured_status_contradiction")
        elif assured in {"false", "0", "no"} and positive_assured and not negative_assured:
            flags.append("possible_assured_status_contradiction")

        if warranty_value.isdigit() and int(warranty_value) > 0:
            # Only compare explicit warranty durations where they are stated.
            duration_matches = re.findall(
                r"\b(\d{1,2})\s*(month|months|month[s]?|year|years|साल|महीने|महीना)\b",
                normalized_answer,
            )
            for number, unit in duration_matches:
                described_months = int(number) * (12 if unit in {"year", "years", "साल"} else 1)

                additive_warranty = re.search(
                    r"\b(additional|in addition to|plus|separate|combined with|along with)\b"
                    r"|अतिरिक्त|के अलावा|अलग से",
                    normalized_answer,
                    flags=re.IGNORECASE,
                )

                if (
                    described_months != int(warranty_value)
                    and "warranty" in normalized_answer
                    and not additive_warranty
                ):

                    flags.append("possible_warranty_duration_mismatch")
                    break

    status = "review_flag" if flags else "catalogue_match_checked"
    return status, " | ".join(dict.fromkeys(notes + flags))


def run_offline_quality_audit(
    practice_questions: list[dict[str, str]],
    step2_rows: dict[str, dict[str, str]],
) -> None:
    """Write detailed deterministic audit flags without calling an LLM."""
    real_ids = load_real_section_ids()
    catalogue = load_product_catalogue()
    audit_rows: list[dict[str, str]] = []

    for question in practice_questions:
        qid = get_value(question, ("id", "question_id"))
        if not qid:
            continue

        result = step2_rows.get(qid, {})
        prompt = get_value(question, ("question",))
        answer = get_value(result, ANSWER_COLUMNS)
        retrieved = {
            normalize_identifier(item)
            for item in extract_section_ids(get_value(result, SECTION_COLUMNS))
        }
        gold_all = {
            normalize_identifier(item)
            for item in extract_section_ids(get_value(question, ("gold_evidence",)))
        }
        gold_sections = gold_all & real_ids
        missing_gold = sorted(gold_sections - retrieved)

        citations = [
            normalize_identifier(item)
            for item in CITATION_PATTERN.findall(answer)
        ]
        citations = list(dict.fromkeys(citations))
        invalid_citations = sorted(set(citations) - real_ids)
        valid_but_not_retrieved = sorted(
            set(citations) & real_ids - retrieved
        )
        valid_and_retrieved = sorted(set(citations) & real_ids & retrieved)

        truncation_flag = answer_looks_truncated(answer)
        expected = get_value(question, ("expected_decision",))
        decision_status, decision_note = decision_signal(answer, expected)
        product_status, product_note = product_consistency_flags(
            prompt, answer, catalogue
        )

        answer_status = get_value(result, STATUS_COLUMNS) or (
            "success" if row_has_answer(result) else "missing"
        )
        if not answer:
            completeness = "empty_or_missing"
        elif truncation_flag:
            completeness = "possible_truncation_review"
        else:
            completeness = "no_obvious_truncation_signal"

        audit_rows.append({
            "question_id": qid,
            "question": prompt,
            "intent": get_value(question, ("intent",)),
            "expected_decision": expected,
            "answer_status": answer_status,
            "provider": get_value(result, PROVIDER_COLUMNS),
            "model": get_value(result, MODEL_COLUMNS),
            "gold_section_ids": "; ".join(sorted(gold_sections)),
            "retrieved_section_ids": "; ".join(sorted(retrieved)),
            "missing_gold_section_ids": "; ".join(missing_gold),
            "answer_citation_ids": "; ".join(citations),
            "valid_citations_retrieved": "; ".join(valid_and_retrieved),
            "valid_citations_not_retrieved": "; ".join(valid_but_not_retrieved),
            "unsupported_or_fabricated_citations": "; ".join(invalid_citations),
            "citation_audit_status": (
                "unsupported_citation_found" if invalid_citations
                else "citation_not_in_retrieved_evidence" if valid_but_not_retrieved
                else "no_bracketed_citation" if not citations
                else "citations_match_retrieved_sections"
            ),
            "response_completeness_flag": completeness,
            "expected_decision_signal": decision_status,
            "decision_audit_note": decision_note,
            "product_catalogue_check": product_status,
            "product_catalogue_notes": product_note,
            "answer": answer,
        })

    fields = [
        "question_id", "question", "intent", "expected_decision",
        "answer_status", "provider", "model", "gold_section_ids",
        "retrieved_section_ids", "missing_gold_section_ids",
        "answer_citation_ids", "valid_citations_retrieved",
        "valid_citations_not_retrieved", "unsupported_or_fabricated_citations",
        "citation_audit_status", "response_completeness_flag",
        "expected_decision_signal", "decision_audit_note",
        "product_catalogue_check", "product_catalogue_notes", "answer",
    ]

    OFFLINE_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OFFLINE_AUDIT_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    citation_invalid = sum(bool(row["unsupported_or_fabricated_citations"]) for row in audit_rows)
    citation_unretrieved = sum(bool(row["valid_citations_not_retrieved"]) for row in audit_rows)
    no_citations = sum(row["citation_audit_status"] == "no_bracketed_citation" for row in audit_rows)
    truncation = sum(row["response_completeness_flag"] == "possible_truncation_review" for row in audit_rows)
    missing_sections = sum(bool(row["missing_gold_section_ids"]) for row in audit_rows)
    product_flags = sum(row["product_catalogue_check"] == "review_flag" for row in audit_rows)
    decision_review = sum(row["expected_decision_signal"] == "review" for row in audit_rows)

    print("\nOFFLINE STEP 2 QUALITY AUDIT (heuristic flags; no LLM calls)")
    print("-" * 60)
    print(f"Responses audited: {len(audit_rows)}")
    print(f"Answers with unsupported/fabricated citation IDs: {citation_invalid}")
    print(f"Answers citing real sections not in retrieval: {citation_unretrieved}")
    print(f"Answers with no bracketed section citation: {no_citations}")
    print(f"Potentially truncated/empty answers to review: {truncation}")
    print(f"Questions still missing expected sections: {missing_sections}")
    print(f"Product catalogue contradictions needing review: {product_flags}")
    print(f"Decision-handling heuristic flags needing review: {decision_review}")
    print(f"Detailed offline audit saved to: {OFFLINE_AUDIT_FILE}")
    print("These are deterministic review flags, not semantic accuracy scores.")


def main() -> None:
    """Compare both stages on the original practice-question set."""
    print("ZentroMart Step 1 vs Step 2 comparison")
    print("=" * 56)

    practice_questions = read_csv(QUESTIONS_FILE)
    step1_rows = index_by_question_id(
        read_csv(STEP1_FILE),
        "Step 1",
    )
    step2_rows = index_by_question_id(
        read_csv(STEP2_FILE),
        "Step 2",
    )

    # The practice dataset defines the comparison set and its order.
    question_ids = [
        get_value(question, ("id", "question_id"))
        for question in practice_questions
    ]

    question_ids = [question_id for question_id in question_ids if question_id]

    if not question_ids:
        raise ValueError("No question IDs found in the practice dataset.")

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate question IDs found in practice/questions.csv.")

    comparison_rows: list[dict[str, str]] = []

    for question in practice_questions:
        question_id = get_value(question, ("id", "question_id"))

        if not question_id:
            continue

        step1 = step1_rows.get(question_id, {})
        step2 = step2_rows.get(question_id, {})

        question_text = get_value(question, ("question",))
        must_include = split_expected_phrases(
            get_value(question, ("must_include",))
        )
        must_not_include = split_expected_phrases(
            get_value(question, ("must_not_include",))
        )
        gold_evidence = get_value(question, ("gold_evidence",))
        gold_ids = extract_section_ids(gold_evidence)

        step1_answer = get_value(step1, ANSWER_COLUMNS)
        step2_answer = get_value(step2, ANSWER_COLUMNS)

        step1_available = row_has_answer(step1)
        step2_available = row_has_answer(step2)

        step1_check = evaluate_checklist(
            answer=step1_answer,
            answer_available=step1_available,
            must_include=must_include,
            must_not_include=must_not_include,
        )

        step2_check = evaluate_checklist(
            answer=step2_answer,
            answer_available=step2_available,
            must_include=must_include,
            must_not_include=must_not_include,
        )

        retrieved_ids = extract_section_ids(
            get_value(step2, SECTION_COLUMNS)
        )

        matched_gold = [
            section_id for section_id in gold_ids
            if section_id in retrieved_ids
        ]

        if gold_ids:
            evidence_recall = percent(
                len(matched_gold),
                len(gold_ids),
            )
            evidence_precision = percent(
                len(matched_gold),
                len(retrieved_ids),
            )
        else:
            evidence_recall = "N/A"
            evidence_precision = "N/A"

        comparison_rows.append({
            "question_id": question_id,
            "question": question_text,
            "expected_decision": get_value(
                question, ("expected_decision",)
            ),
            "gold_evidence_ids": format_list(gold_ids),
            "must_include": format_list(must_include),
            "must_not_include": format_list(must_not_include),

            "step1_status": get_value(step1, STATUS_COLUMNS)
            or ("success" if step1_available else "missing"),
            "step1_provider": get_value(step1, PROVIDER_COLUMNS),
            "step1_model": get_value(step1, MODEL_COLUMNS),
            "step1_answer_available": yes_no(step1_available),
            "step1_required_phrases_matched": str(
                step1_check["required_matched"]
            ),
            "step1_required_phrases_total": str(
                step1_check["required_total"]
            ),
            "step1_missing_required_phrases": format_list(
                step1_check["missing_required"]
            ),
            "step1_forbidden_phrase_hits": format_list(
                step1_check["forbidden_hits"]
            ),
            "step1_checklist_pass": (
                "N/A"
                if step1_check["checklist_pass"] is None
                else yes_no(step1_check["checklist_pass"])
            ),
            "step1_answer": step1_answer,

            "step2_status": get_value(step2, STATUS_COLUMNS)
            or ("success" if step2_available else "missing"),
            "step2_provider": get_value(step2, PROVIDER_COLUMNS),
            "step2_model": get_value(step2, MODEL_COLUMNS),
            "step2_answer_available": yes_no(step2_available),
            "step2_required_phrases_matched": str(
                step2_check["required_matched"]
            ),
            "step2_required_phrases_total": str(
                step2_check["required_total"]
            ),
            "step2_missing_required_phrases": format_list(
                step2_check["missing_required"]
            ),
            "step2_forbidden_phrase_hits": format_list(
                step2_check["forbidden_hits"]
            ),
            "step2_checklist_pass": (
                "N/A"
                if step2_check["checklist_pass"] is None
                else yes_no(step2_check["checklist_pass"])
            ),
            "step2_answer": step2_answer,

            "step2_retrieved_section_ids": format_list(retrieved_ids),
            "step2_gold_evidence_recall": evidence_recall,
            "step2_gold_evidence_precision": evidence_precision,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
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
        writer.writerows(comparison_rows)

    print(f"Questions in practice set: {len(question_ids)}")
    print(f"Questions compared: {len(comparison_rows)}")
    print(f"Questions missing from Step 1: {sum(not row['step1_answer_available'] == 'yes' for row in comparison_rows)}")
    print(f"Questions missing from Step 2: {sum(not row['step2_answer_available'] == 'yes' for row in comparison_rows)}")

    summarize_stage("Step1", comparison_rows)
    summarize_stage("Step2", comparison_rows)

    evidence_rows = [
        row for row in comparison_rows
        if row["step2_gold_evidence_recall"] != "N/A"
    ]

    evidence_recall_values = [
        float(row["step2_gold_evidence_recall"].rstrip("%"))
        for row in evidence_rows
    ]

    if evidence_recall_values:
        mean_recall = sum(evidence_recall_values) / len(
            evidence_recall_values
        )
        print("\nStep 2 retrieval")
        print("-" * 44)
        print(
            f"Questions with identifiable gold evidence: "
            f"{len(evidence_rows)}"
        )
        print(
            f"Mean gold-evidence recall: {mean_recall:.2f}%"
        )

    print(f"\nComparison CSV saved to:\n{OUTPUT_FILE}")
    print(
        "\nInterpretation note: phrase checks and evidence overlap are "
        "diagnostic metrics, not semantic accuracy. Review the generated "
        "answers for factual correctness, decision handling, and citation "
        "quality before claiming an accuracy improvement."
    )

    # Add a second, fully offline answer-quality audit while preserving the
    # existing Step 1 vs Step 2 comparison output and metrics.
    run_offline_quality_audit(practice_questions, step2_rows)


if __name__ == "__main__":
    main()
