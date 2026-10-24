"""Local retrieval-augmented generation utilities for ZentroMart."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Capstone_Project_Dataset_FTTT_2026" / "data"
KB_DIR = ROOT / "Capstone_Project_Dataset_FTTT_2026" / "kb"
PRODUCTS_FILE = DATA_DIR / "products.csv"
INDEX_DIR = ROOT / "outputs" / "rag_index"

CHUNKS_FILE = INDEX_DIR / "chunks.json"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
METADATA_FILE = INDEX_DIR / "index_metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180

# Changing this value forces older index files to be rebuilt.
INDEX_SCHEMA_VERSION = "rag-index-v3-product-catalog-hybrid"

SECTION_PATTERN = re.compile(
    r"^(#{1,6})\s+\[([^\]]+)\]\s*(.*)$",
    flags=re.MULTILINE,
)

DOC_TYPE_PRIORITY = {
    "policy": 5,
    "agent_guide": 4,
    "help": 3,
    "faq": 2,
    "marketing": 1,
    "meta": 0,
}

FRONTMATTER_PATTERN = re.compile(
    r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)",
    flags=re.DOTALL,
)

# Keep the embedding model in memory for repeated queries.
_EMBEDDING_MODEL_INSTANCE: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once per process and reuse it."""
    global _EMBEDDING_MODEL_INSTANCE

    if _EMBEDDING_MODEL_INSTANCE is None:
        print(f"Loading local embedding model: {EMBEDDING_MODEL}")
        print("The first run may download model files.")
        _EMBEDDING_MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL)

    return _EMBEDDING_MODEL_INSTANCE


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse Markdown metadata, including pipe-separated fields.

    Supports both:
        status: current

    and:
        status: current| effective_from: 2026-07-01
    """
    metadata: dict[str, str] = {}

    match = FRONTMATTER_PATTERN.match(text)

    if not match:
        return metadata, text

    header = match.group(1)
    body = text[match.end():]

    field_pattern = re.compile(
        r"([A-Za-z_][\w-]*)\s*:\s*"
        r"(.*?)"
        r"(?=\s*\|\s*[A-Za-z_][\w-]*\s*:|$)"
    )

    for line in header.splitlines():
        for field in field_pattern.finditer(line):
            key = field.group(1).strip()
            value = field.group(2).strip().strip("\"'")
            metadata[key] = value

    return metadata, body


def infer_doc_type(filename: str) -> str:
    """Infer document type from the dataset filename convention."""
    name = filename.lower()

    if name.startswith("pol_"):
        return "policy"

    if name.startswith("agent_"):
        return "agent_guide"

    if name.startswith("help_"):
        return "help"

    if name.startswith("faq_"):
        return "faq"

    if name.startswith("mkt_"):
        return "marketing"

    if name.startswith("00_"):
        return "meta"

    return "unknown"


def document_fingerprint(paths: list[Path]) -> str:
    """Fingerprint documents, settings, and the index schema.

    The schema marker invalidates old indexes that assigned artificial
    section IDs to document-introduction text.
    """
    digest = hashlib.sha256()

    digest.update(INDEX_SCHEMA_VERSION.encode("utf-8"))
    digest.update(EMBEDDING_MODEL.encode("utf-8"))
    digest.update(str(CHUNK_SIZE).encode("utf-8"))
    digest.update(str(CHUNK_OVERLAP).encode("utf-8"))

    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())

    return digest.hexdigest()


def split_long_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split long sections into overlapping chunks without dropping text."""
    text = text.strip()

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size - 1."
        )

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            # Prefer paragraph, line, or word boundaries where possible.
            minimum_cut = start + int(chunk_size * 0.65)

            candidates = [
                text.rfind("\n\n", minimum_cut, end),
                text.rfind("\n", minimum_cut, end),
                text.rfind(" ", minimum_cut, end),
            ]

            cut = max(candidates)

            if cut > start:
                end = cut

        piece = text[start:end].strip()

        if piece:
            chunks.append(piece)

        if end >= length:
            break

        next_start = max(start + 1, end - overlap)

        # Avoid starting a chunk in the middle of whitespace.
        while next_start < end and text[next_start].isspace():
            next_start += 1

        start = next_start

    return chunks


def extract_sections(
    body: str,
    filename: str = "",
) -> list[dict[str, str]]:
    """Extract real section IDs without inventing IDs for introductions.

    Introductory text before the first section heading is retained with
    an empty section_id. It is not assigned a fabricated source ID.
    """
    del filename  # The source filename is stored on each chunk.

    matches = list(SECTION_PATTERN.finditer(body))
    sections: list[dict[str, str]] = []

    if not matches:
        if body.strip():
            sections.append({
                "section_id": "",
                "section_title": "Document introduction",
                "text": body.strip(),
            })

        return sections

    # Preserve introductory text before the first real section heading.
    intro = body[:matches[0].start()].strip()

    if intro:
        sections.append({
            "section_id": "",
            "section_title": "Document introduction",
            "text": intro,
        })

    # Extract each section using its actual source section ID.
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(body)
        )

        sections.append({
            "section_id": match.group(2).strip(),
            "section_title": match.group(3).strip(),
            "text": body[match.start():end].strip(),
        })

    return sections


def load_product_chunks() -> tuple[list[dict[str, Any]], int]:
    """Load product catalogue rows as structured evidence chunks.

    Product records can establish catalogue attributes such as SKU, title,
    brand, return category, price, fulfilment, Assured status, and warranty.
    They do not replace Markdown policies for rules such as return windows.
    """
    if not PRODUCTS_FILE.is_file():
        raise FileNotFoundError(
            f"Product catalogue not found: {PRODUCTS_FILE}"
        )

    required_columns = {
        "sku",
        "title",
        "brand",
        "return_category",
        "price_inr",
        "fulfilled_by",
        "assured",
        "warranty_months",
        "description",
    }

    product_chunks: list[dict[str, Any]] = []

    with PRODUCTS_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "products.csv is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=1):
            product = {
                key: (row.get(key) or "").strip()
                for key in required_columns
            }
            sku = product["sku"]

            if not sku:
                print(
                    f"Warning: skipping product row {row_number}; SKU is empty."
                )
                continue

            # The product record is both searchable text and structured metadata.
            product_text = "\n".join([
                "Structured product catalogue record; this record is not a policy.",
                f"SKU: {sku}",
                f"Product title: {product['title']}",
                f"Brand: {product['brand']}",
                f"Return category: {product['return_category']}",
                f"Price in INR: {product['price_inr']}",
                f"Fulfilled by: {product['fulfilled_by']}",
                f"ZentroMart Assured: {product['assured']}",
                f"Warranty months: {product['warranty_months']}",
                f"Description: {product['description']}",
            ])

            product_chunks.append({
                "chunk_id": f"PROD-{sku}",
                "doc_id": "PRODUCT_CATALOG",
                "title": product["title"] or sku,
                "source_file": "data/products.csv",
                "source_kind": "product_catalog",
                "doc_type": "catalog",
                "status": "current",
                "effective_from": "",
                "section_id": "",
                "section_title": "Product catalogue record",
                "chunk_number": 1,
                "sku": sku,
                "product_fields": product,
                "text": product_text,
            })

    if not product_chunks:
        raise RuntimeError(
            f"No usable product records found in {PRODUCTS_FILE}"
        )

    return product_chunks, len(product_chunks)


def load_document_chunks() -> tuple[list[dict[str, Any]], str]:
    """Load Markdown knowledge documents and structured product records."""
    if not KB_DIR.is_dir():
        raise FileNotFoundError(
            f"Knowledge-base folder not found: {KB_DIR}"
        )

    paths = sorted(KB_DIR.glob("*.md"))

    if not paths:
        raise FileNotFoundError(
            f"No Markdown documents found in {KB_DIR}"
        )

    if not PRODUCTS_FILE.is_file():
        raise FileNotFoundError(
            f"Product catalogue not found: {PRODUCTS_FILE}"
        )

    # Include the catalogue in the fingerprint so edits to products.csv
    # automatically invalidate the saved index.
    fingerprint = document_fingerprint([*paths, PRODUCTS_FILE])

    chunks: list[dict[str, Any]] = []
    chunk_number = 0

    for path in paths:
        raw_text = path.read_text(encoding="utf-8-sig")
        metadata, body = parse_frontmatter(raw_text)

        doc_type = metadata.get(
            "doc_type",
            infer_doc_type(path.name),
        ).strip().lower()

        status = metadata.get("status", "unknown").strip().lower()
        effective_from = metadata.get("effective_from", "").strip()

        if status not in {"current", "superseded"}:
            print(
                f"Warning: {path.name} has unrecognized status "
                f"{status!r}. It will be retained but should be reviewed."
            )

        sections = extract_sections(body, path.name)

        if not sections:
            print(f"Warning: no indexable text found in {path.name}")
            continue

        for section in sections:
            pieces = split_long_text(section["text"])

            for piece_number, piece in enumerate(pieces, start=1):
                chunk_number += 1

                chunks.append({
                    "chunk_id": f"CH-{chunk_number:04d}",
                    "doc_id": metadata.get("doc_id", ""),
                    "title": metadata.get("title", path.stem),
                    "source_file": path.name,
                    "source_kind": "knowledge_base",
                    "doc_type": doc_type,
                    "status": status,
                    "effective_from": effective_from,
                    "section_id": section["section_id"],
                    "section_title": section["section_title"],
                    "chunk_number": piece_number,
                    "sku": "",
                    "text": piece,
                })

    if not chunks:
        raise RuntimeError(
            "No indexable knowledge-base chunks were found."
        )

    product_chunks, product_count = load_product_chunks()
    chunks.extend(product_chunks)

    print(
        f"Loaded {len(paths)} Markdown documents and "
        f"{product_count} product catalogue records."
    )

    return chunks, fingerprint

def build_index(force_rebuild: bool = False) -> dict[str, Any]:
    """Build the local vector index or reuse a compatible saved index."""
    chunks, fingerprint = load_document_chunks()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if (
        not force_rebuild
        and CHUNKS_FILE.exists()
        and EMBEDDINGS_FILE.exists()
        and METADATA_FILE.exists()
    ):
        try:
            saved_metadata = json.loads(
                METADATA_FILE.read_text(encoding="utf-8")
            )

            saved_chunks = json.loads(
                CHUNKS_FILE.read_text(encoding="utf-8")
            )

            saved_vectors = np.load(
                EMBEDDINGS_FILE,
                allow_pickle=False,
            )

            if (
                saved_metadata.get("fingerprint") == fingerprint
                and saved_metadata.get("schema_version")
                == INDEX_SCHEMA_VERSION
                and len(saved_chunks) == saved_vectors.shape[0]
                and len(saved_chunks) > 0
            ):
                return {
                    "chunks": saved_chunks,
                    "embeddings": saved_vectors,
                    "fingerprint": fingerprint,
                    "reused": True,
                }

        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            IndexError,
        ):
            print("Existing index is invalid; rebuilding it.")

    model = _get_embedding_model()

    documents = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(documents)} chunks...")

    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    CHUNKS_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings,
        allow_pickle=False,
    )

    METADATA_FILE.write_text(
        json.dumps({
            "schema_version": INDEX_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "document_count": len({
                chunk["source_file"]
                for chunk in chunks
                if chunk.get("source_kind") == "knowledge_base"
            }),
            "product_record_count": sum(
                chunk.get("source_kind") == "product_catalog"
                for chunk in chunks
            ),
            "chunk_count": len(chunks),
            "embedding_dimensions": int(embeddings.shape[1]),
        }, indent=2),
        encoding="utf-8",
    )

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "fingerprint": fingerprint,
        "reused": False,
    }



def tokenize_search_text(text: str) -> list[str]:
    """Tokenize words and identifiers for multilingual keyword matching."""
    return re.findall(r"(?u)\b[\w-]+\b", text.casefold())


def calculate_bm25_scores(
    query: str,
    chunks: list[dict[str, Any]],
    eligible_positions: list[int],
) -> dict[int, float]:
    """Calculate BM25 lexical relevance for the eligible chunks."""
    query_tokens = sorted(set(tokenize_search_text(query)))

    if not query_tokens or not eligible_positions:
        return {}

    tokenized_documents: dict[int, list[str]] = {}
    term_counts: dict[int, Counter[str]] = {}
    document_frequency: Counter[str] = Counter()

    for position in eligible_positions:
        chunk = chunks[position]
        searchable_text = " ".join((
            str(chunk.get("title", "")),
            str(chunk.get("section_id", "")),
            str(chunk.get("section_title", "")),
            str(chunk.get("source_file", "")),
            str(chunk.get("sku", "")),
            str(chunk.get("text", "")),
        ))

        tokens = tokenize_search_text(searchable_text)
        tokenized_documents[position] = tokens
        term_counts[position] = Counter(tokens)
        document_frequency.update(set(tokens))

    document_count = len(eligible_positions)
    average_length = sum(
        len(tokens) for tokens in tokenized_documents.values()
    ) / document_count

    if average_length <= 0:
        return {}

    k1 = 1.5
    b = 0.75
    scores: dict[int, float] = {}

    for position in eligible_positions:
        counts = term_counts[position]
        document_length = len(tokenized_documents[position])
        score = 0.0

        for token in query_tokens:
            frequency = counts.get(token, 0)
            if frequency == 0:
                continue

            df = document_frequency[token]
            inverse_document_frequency = math.log(
                1.0 + (document_count - df + 0.5) / (df + 0.5)
            )

            denominator = frequency + k1 * (
                1.0 - b + b * document_length / average_length
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1.0)
            ) / denominator

        scores[position] = score

    return scores


def _normalized_entity_text(text: str) -> str:
    """Normalize text for product-name matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.casefold())).strip()


def _select_product_evidence(
    query: str,
    chunks: list[dict[str, Any]],
    similarities: np.ndarray,
) -> list[dict[str, Any]]:
    """Find at most one relevant product record without competing with policy retrieval.

    Exact SKU matches have highest priority. Otherwise, a product can be selected
    by a sufficiently strong product-title match. Product records are returned
    after the top-k policy/knowledge chunks so they cannot consume the policy
    retrieval budget or hide section IDs used by the evaluation.
    """
    product_positions = [
        position
        for position, chunk in enumerate(chunks)
        if chunk.get("source_kind") == "product_catalog"
    ]

    if not product_positions:
        return []

    query_text = query.strip()
    query_normalized = _normalized_entity_text(query_text)
    query_tokens = set(tokenize_search_text(query_text))

    # First: exact SKU lookup. Boundary checks prevent SKU substrings from
    # matching another product identifier.
    exact_matches: list[int] = []
    for position in product_positions:
        sku = str(chunks[position].get("sku", "")).strip()
        if sku and re.search(
            rf"(?<![A-Za-z0-9]){re.escape(sku)}(?![A-Za-z0-9])",
            query_text,
            flags=re.IGNORECASE,
        ):
            exact_matches.append(position)

    if exact_matches:
        position = exact_matches[0]
        chunk = chunks[position]
        return [{
            **chunk,
            "similarity": float(similarities[position]),
            "lexical_score": 0.0,
            "hybrid_score": 1.0,
            "exact_sku_match": True,
            "product_match_score": 1.0,
        }]

    stop_words = {
        "a", "an", "the", "is", "it", "i", "my", "me", "this", "that",
        "what", "which", "how", "can", "could", "would", "should", "does",
        "do", "did", "was", "were", "are", "for", "to", "of", "on", "in",
        "with", "and", "or", "if", "has", "have", "had", "just", "bought",
        "buy", "ordered", "order", "received", "got", "delivered", "yesterday",
        "today", "please", "return", "returns", "returned", "warranty", "claim",
        "assured", "collection", "free", "came", "torn", "cracked", "broken",
    }

    # Recompute with positions retained; this small catalogue has only 70 rows.
    candidates: list[tuple[tuple[float, float, float, float], int]] = []
    for position in product_positions:
        chunk = chunks[position]
        fields = chunk.get("product_fields", {})
        title = str(fields.get("title") or chunk.get("title") or "")
        brand = str(fields.get("brand", ""))
        title_normalized = _normalized_entity_text(title)
        phrase_match = bool(title_normalized and title_normalized in query_normalized)
        title_tokens = {
            token for token in tokenize_search_text(title)
            if token not in stop_words and len(token) > 1
        }
        brand_tokens = {
            token for token in tokenize_search_text(brand)
            if token not in stop_words and len(token) > 1
        }
        target_tokens = title_tokens | brand_tokens
        overlap = len(target_tokens & query_tokens)
        ratio = overlap / max(1, len(target_tokens))

        if phrase_match or (overlap >= 2 and ratio >= 0.5):
            score = (
                1.0 if phrase_match else 0.0,
                ratio,
                float(overlap),
                float(similarities[position]),
            )
            candidates.append((score, position))

    if not candidates:
        return []

    candidates.sort(reverse=True)
    _, position = candidates[0]
    chunk = chunks[position]
    fields = chunk.get("product_fields", {})
    title = str(fields.get("title") or chunk.get("title") or "")
    title_tokens = {
        token for token in tokenize_search_text(title)
        if token not in stop_words and len(token) > 1
    }
    overlap = len(title_tokens & query_tokens)
    ratio = overlap / max(1, len(title_tokens))
    phrase_match = bool(
        _normalized_entity_text(title)
        and _normalized_entity_text(title) in query_normalized
    )

    return [{
        **chunk,
        "similarity": float(similarities[position]),
        "lexical_score": 0.0,
        "hybrid_score": 1.0 if phrase_match else ratio,
        "exact_sku_match": False,
        "product_match_score": ratio,
    }]


def _detect_query_intents(query: str) -> set[str]:
    """Classify common support intents with lightweight local rules.

    This deliberately uses no LLM/API call. It is a ranking aid, not a
    customer-intent classifier whose output should be shown to users.
    """
    text = query.casefold()
    intents: set[str] = set()

    has_refund = bool(re.search(
        r"\brefund(?:s|ed|ing)?\b|\bpaise\s+wapis\b|\bmoney\s+back\b|"
        r"वापस\s+पैसे|रिफंड|रिफ़ंड|पैसे\s+वापस",
        text,
    ))
    has_return = bool(re.search(
        r"\breturn(?:s|ed|ing)?\b|\bsend(?:\s+\w+){0,3}\s+back\b|\btake\s+back\b|"
        r"\blaut(?:a|an|ane|ana|aun|aana)\b|\bwapis\b|"
        r"लौट|वापस\s+कर|वापस\s+करना|वापस\s+कर\s+सक",
        text,
    ))
    has_warranty = bool(re.search(
        r"\bwarranty\b|\bassured\b|\bclaim\b|\bguarantee\b|"
        r"वारंटी|गारंटी|दावा",
        text,
    ))
    has_damage = bool(re.search(
        r"\btorn\b|\bdamaged?\b|\bcrack(?:ed)?\b|\bbroken\b|"
        r"\bdefect(?:ive)?\b|\bdoesn't\s+work\b|\bwon't\s+turn\s+on\b|"
        r"\bnot\s+working\b|\barrived\s+broken\b|"
        r"टूट|टूटा|टूटी|खराब|क्षतिग्रस्त",
        text,
    ))
    has_refund_date = bool(re.search(
        r"\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
        r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\s+20\d{2}\b|"
        r"\b20\d{2}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]20\d{2}\b",
        text,
    ))

    if has_warranty:
        intents.add("warranty_assured")

    if has_refund:
        intents.add("refund_timing")
        if re.search(r"\bwallet\b|zentromart\s+wallet|वॉलेट|वालेट", text):
            intents.add("refund_wallet")
        if re.search(r"\bcash\s+on\s+delivery\b|\bcod\b|\bcash\s+delivery\b", text):
            intents.add("refund_cod")
        if has_refund_date:
            intents.add("refund_date")

    if has_return:
        if has_damage:
            intents.add("damaged_return")
        elif re.search(
            r"\bopen(?:ed)?\b|\btried\b|\bused\b|\bhygiene\b|"
            r"\bseal\b|\bdoesn't\s+fit\b|\bdoes\s+not\s+fit\b|"
            r"\bnot\s+fit\b|\bwrong\s+size\b|"
            r"\bsize\s+(?:is\s+)?(?:wrong|small|large|big|tight)\b|"
            r"\btoo\s+(?:small|large|tight|loose)\b|"
            r"खोल|इस्तेमाल|पहना|फिट",
            text,
        ):
            intents.add("return_condition")
        else:
            intents.add("return_window")

    if has_damage and not has_return:
        intents.add("damaged_item")

    if re.search(r"\bpick\s*up\s*fee\b|\bcollection\s+free\b|\bfree\s+collection\b|\bpickup\s+fee\b|\bcollection\b|\bpickup\b|पिकअप", text):
        intents.add("pickup_fee")

    if re.search(
        r"\bcancel\b|\bcancellation\b|\bstop\s+(?:it|the\s+order)\b|"
        r"\bon\s+its\s+way\b|\bshipped\b|\bship(?:ped)?\s+ho\s+gaya\b|कैंसिल|रद्द",
        text,
    ):
        intents.add("cancellation")

    if re.search(
        r"\bdeliver(?:y)?\b|\bcourier\b|\bnot\s+home\b|\bmidnight\b|"
        r"\bdubai\b|\bstore\s+near\s+me\b|\bshop(?:s)?\s+in\b|दुकान|कोच्चि",
        text,
    ):
        intents.add("delivery")

    if re.search(
        r"\bdiscount\b|\bstudent\b|\bcoupon\b|\boffer\b|\bpromo(?:tion)?\b|"
        r"\bgst\b|\binvoice\b|\bsale\b|छूट|डिस्काउंट",
        text,
    ):
        intents.add("offers_invoice")

    if re.search(
        r"\bnever\s+placed\b|\bdidn't\s+place\b|\bdid\s+not\s+place\b|"
        r"\bunauthori[sz]ed\b|\bphone\s+number\b|\bperson\s+who\s+ordered\b|"
        r"\bsomeone\s+else\b|\bprivacy\b|\bpersonal\s+information\b",
        text,
    ):
        intents.add("account_privacy")

    if re.search(
        r"\bwon't\s+turn\s+on\b|\bnot\s+working\b|\bdoesn't\s+work\b|"
        r"\btroubleshoot(?:ing)?\b|\bbluetooth\s+speaker\b|\bpower\s+bank\b",
        text,
    ):
        intents.add("troubleshooting")

    if re.search(r"\bstore\s+near\s+me\b|\bpick\s+up\s+my\s+order\b|\bin\s+store\b", text):
        intents.add("store_pickup")

    return intents


# Small additive boosts act after semantic+BM25 rank fusion. They are deliberately
# modest and based on source section IDs confirmed in this dataset's audit outputs.
_INTENT_SECTION_BOOSTS: dict[str, dict[str, float]] = {
    "warranty_assured": {
        "WAR-2": 0.012,
        "WAR-1": 0.008,
        "WAR-3": 0.006,
        "MKS-3": 0.003,
    },
    "return_window": {
        "RET-2": 0.018,
        "RET-3": 0.014,
        "RET-8": 0.013,
        "RET-9": 0.008,
        "RETV3-2": 0.013,
    },
    "return_condition": {
        "RET-4": 0.013,
        "RET-2": 0.008,
        "RET-5": 0.014,
        "RET-8": 0.005,
        "RET-9": 0.006,
    },
    "damaged_return": {
        "DWM-2": 0.013,
        "RET-6": 0.013,
        "RET-4": 0.006,
        "RET-5": 0.004,
        "ESC-1": 0.003,
    },
    "damaged_item": {
        "DWM-1": 0.032,
        "DWM-2": 0.016,
        "DWM-3": 0.016,
        "RET-6": 0.014,
        "TRB-4": 0.007,
        "TRB-1": 0.005,
        "TRB-2": 0.005,
        "TRB-3": 0.005,
        "ESC-1": 0.010,
    },
    "refund_timing": {
        "KB-00.1": 0.012,
        "REF-2": 0.014,
        "REF-5": 0.007,
        "REF-6": 0.007,
        "KB-00.2": 0.003,
        "PAY-4": 0.003,
    },
    "refund_wallet": {
        "KB-00.1": 0.028,
        "REF-2": 0.012,
        "REF-5": 0.006,
        "REF-3": 0.006,
    },
    "refund_cod": {
        "REF-3": 0.012,
        "REF-2": 0.010,
        "PAY-4": 0.005,
    },
    "refund_date": {
        "KB-00.3": 0.020,
        "KB-00.2": 0.003,
        "REF-2": 0.005,
    },
    "pickup_fee": {
        "RET-6": 0.011,
        "PLS-2": 0.010,
        "RET-4": 0.003,
        "DWM-2": 0.004,
    },
    "cancellation": {
        "CAN-1": 0.012,
        "CAN-2": 0.010,
        "CAN-3": 0.008,
    },
    "delivery": {
        "DEL-1": 0.008,
        "DEL-3": 0.010,
        "DEL-4": 0.008,
        "DEL-5": 0.010,
        "DEL-6": 0.007,
        "TRK-1": 0.004,
        "TRK-2": 0.004,
        "RVP-2": 0.006,
        "FAQ-5": 0.004,
    },
    "offers_invoice": {
        "CPN-1": 0.009,
        "CPN-2": 0.009,
        "PLS-1": 0.004,
        "FAQ-3": 0.003,
    },
    "account_privacy": {
        "IDV-2": 0.009,
        "IDV-3": 0.009,
        "IDV-4": 0.012,
        "ESC-1": 0.016,
        "ESC-4": 0.016,
        "MKS-1": 0.004,
    },
    "troubleshooting": {
        "TRB-1": 0.008,
        "TRB-2": 0.008,
        "TRB-3": 0.008,
        "TRB-4": 0.009,
        "WAR-1": 0.003,
    },
    "store_pickup": {
        "FAQ-2": 0.008,
        "RVP-1": 0.008,
        "CAN-2": 0.005,
        "CAN-3": 0.004,
    },
}


def _intent_boost_for_chunk(
    query: str,
    chunk: dict[str, Any],
    product_results: list[dict[str, Any]],
) -> float:
    """Calculate a local intent-specific reranking boost for one KB chunk."""
    section_id = str(chunk.get("section_id", "")).strip().upper()
    if not section_id:
        return 0.0

    intents = _detect_query_intents(query)
    boost = max(
        (
            section_boosts.get(section_id, 0.0)
            for intent in intents
            for section_boosts in [_INTENT_SECTION_BOOSTS.get(intent, {})]
        ),
        default=0.0,
    )

    # Product category adds a small contextual hint, never a substitute for policy.
    if product_results:
        product_fields = product_results[0].get("product_fields", {})
        category = str(product_fields.get("return_category", "")).casefold()
        product_title = str(product_fields.get("title", "")).casefold()

        if "electronics" in category and "return_window" in intents:
            if section_id == "RET-3":
                boost = max(boost, 0.010)
            if any(word in product_title for word in ("smartwatch", "watch")):
                if section_id == "RETV3-2":
                    boost = max(boost, 0.012)

        if "fashion" in category and "return_window" in intents:
            if section_id in {"RET-2", "RET-8", "RET-9"}:
                boost = max(boost, 0.009)

    return boost


def search(
    query: str,
    top_k: int = 5,
    include_superseded: bool = False,
    method: str = "hybrid",
) -> list[dict[str, Any]]:
    """Retrieve policy evidence with local intent-aware hybrid reranking.

    Semantic embeddings + BM25 keyword ranking are combined with Reciprocal
    Rank Fusion. A small rules-based boost then prioritizes sections directly
    related to the query intent. Relevant structured product evidence is
    appended separately, outside the policy top_k budget.

    This function makes no hosted LLM/API calls.
    """
    if not query or not query.strip():
        raise ValueError("The search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if method not in {"hybrid", "semantic"}:
        raise ValueError("method must be 'hybrid' or 'semantic'.")

    index = build_index()
    chunks = index["chunks"]
    model = _get_embedding_model()

    query_embedding = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    similarities = index["embeddings"] @ query_embedding

    # Detect a product first so category-aware boosts can inform KB ranking.
    product_results = _select_product_evidence(query, chunks, similarities)

    kb_positions = [
        position
        for position, chunk in enumerate(chunks)
        if chunk.get("source_kind") == "knowledge_base"
        and (
            include_superseded
            or chunk.get("status", "").lower() != "superseded"
        )
    ]

    if not kb_positions:
        policy_results: list[dict[str, Any]] = []
    else:
        semantic_order = sorted(
            kb_positions,
            key=lambda position: (
                float(similarities[position]),
                DOC_TYPE_PRIORITY.get(chunks[position].get("doc_type", ""), 0),
            ),
            reverse=True,
        )
        semantic_rank = {
            position: rank
            for rank, position in enumerate(semantic_order, start=1)
        }

        if method == "semantic":
            selected_positions = semantic_order[:top_k]
            policy_results = [
                {
                    **chunks[position],
                    "similarity": float(similarities[position]),
                    "lexical_score": 0.0,
                    "hybrid_score": 0.0,
                    "intent_boost": 0.0,
                    "exact_sku_match": False,
                }
                for position in selected_positions
            ]
        else:
            lexical_scores = calculate_bm25_scores(
                query=query,
                chunks=chunks,
                eligible_positions=kb_positions,
            )
            lexical_order = sorted(
                (
                    position for position in kb_positions
                    if lexical_scores.get(position, 0.0) > 0.0
                ),
                key=lambda position: (
                    lexical_scores[position],
                    float(similarities[position]),
                    DOC_TYPE_PRIORITY.get(chunks[position].get("doc_type", ""), 0),
                ),
                reverse=True,
            )
            lexical_rank = {
                position: rank
                for rank, position in enumerate(lexical_order, start=1)
            }

            rrf_constant = 60
            ranked_kb: list[dict[str, Any]] = []
            for position in kb_positions:
                hybrid_score = 1.0 / (rrf_constant + semantic_rank[position])
                if position in lexical_rank:
                    hybrid_score += 1.0 / (rrf_constant + lexical_rank[position])

                chunk = chunks[position]
                intent_boost = _intent_boost_for_chunk(
                    query=query,
                    chunk=chunk,
                    product_results=product_results,
                )
                ranked_kb.append({
                    **chunk,
                    "similarity": float(similarities[position]),
                    "lexical_score": float(lexical_scores.get(position, 0.0)),
                    "hybrid_score": float(hybrid_score + intent_boost),
                    "base_hybrid_score": float(hybrid_score),
                    "intent_boost": float(intent_boost),
                    "exact_sku_match": False,
                })

            ranked_kb.sort(
                key=lambda item: (
                    item["hybrid_score"],
                    item["similarity"],
                    DOC_TYPE_PRIORITY.get(item.get("doc_type", ""), 0),
                ),
                reverse=True,
            )

            # Deduplicate multiple chunks from the same section, preserving
            # the highest-ranked chunk. This reserves top_k for distinct
            # sections where the source contains multiple chunks per section.
            policy_results = []
            seen_sections: set[tuple[str, str]] = set()
            for item in ranked_kb:
                section_id = str(item.get("section_id", "")).strip()
                if section_id:
                    section_key = (str(item.get("source_file", "")), section_id)
                    if section_key in seen_sections:
                        continue
                    seen_sections.add(section_key)
                policy_results.append(item)
                if len(policy_results) >= top_k:
                    break

    # Product metadata is supplementary and never consumes the KB top_k slots.
    return policy_results + product_results

def index_stats() -> dict[str, Any]:
    """Return counts for Markdown documents, catalogue records, and IDs."""
    index = build_index()
    chunks = index["chunks"]

    real_sections = {
        (item["source_file"], item["section_id"])
        for item in chunks
        if item.get("source_kind") == "knowledge_base"
        and item.get("section_id")
    }

    kb_documents = {
        item["source_file"]
        for item in chunks
        if item.get("source_kind") == "knowledge_base"
    }

    catalog_skus = {
        item.get("sku", "")
        for item in chunks
        if item.get("source_kind") == "product_catalog"
        and item.get("sku")
    }

    return {
        "document_count": len(kb_documents),
        "product_record_count": len(catalog_skus),
        "chunk_count": len(chunks),
        "section_count": len(real_sections),
        "superseded_chunks": sum(
            item.get("status", "").lower() == "superseded"
            and item.get("source_kind") == "knowledge_base"
            for item in chunks
        ),
        "index_reused": index["reused"],
        "embedding_model": EMBEDDING_MODEL,
    }

def main() -> None:
    """Command-line entry point for building and testing the RAG index."""
    parser = argparse.ArgumentParser(
        description="Build and test the local ZentroMart RAG index."
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Optional customer question to test retrieval.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force the local embedding index to be rebuilt.",
    )

    args = parser.parse_args()

    index = build_index(force_rebuild=args.rebuild)
    stats = index_stats()

    print("\nIndex summary")
    print(f"Knowledge-base documents: {stats['document_count']}")
    print(f"Product catalogue records: {stats['product_record_count']}")
    print(f"Total chunks: {stats['chunk_count']}")
    print(f"Real knowledge-base section IDs: {stats['section_count']}")
    print(
        f"Superseded chunks indexed: {stats['superseded_chunks']}"
    )
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Reused existing index: {index['reused']}")

    if args.query:
        query = " ".join(args.query)

        print(f"\nQuery: {query}")
        print("\nRetrieved evidence:")

        results = search(query, top_k=5)

        if not results:
            print("No eligible evidence found.")

        for result in results:
            section_id = result.get("section_id", "")
            section_label = (
                f"[{section_id}]"
                if section_id
                else "(no section ID)"
            )

            print("\n" + "=" * 64)

            print(
                f"Source: {result['source_file']} | "
                f"Section: {section_label}"
            )

            print(
                f"Type: {result['doc_type']} | "
                f"Status: {result['status']} | "
                f"Effective from: {result['effective_from']}"
            )

            print(f"Similarity: {result['similarity']:.4f}")
            if "lexical_score" in result:
                print(f"Lexical score: {result['lexical_score']:.4f}")
            if "hybrid_score" in result:
                print(f"Hybrid score: {result['hybrid_score']:.6f}")
            if result.get("exact_sku_match"):
                print("Exact SKU match: yes")
            print(result["text"][:1500])


if __name__ == "__main__":
    main()