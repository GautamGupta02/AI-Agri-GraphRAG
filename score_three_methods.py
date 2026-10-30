#!/usr/bin/env python3
"""
Shared scorer for standardized AI_Agri_GraphRAG development evaluation.

Methods:
  - No-RAG
  - VectorRAG
  - GraphRAG

Reads model_eval_dev_v2 predictions and the 30-case development manifest.
Uses normalized exact-match scoring, with a special handling rule for answers
that contain only an option number (e.g. "Answer: 4"): the number is resolved
against that example's options from the manifest.

This script does NOT run the model, alter predictions, modify indexes, or read
the reserved final-test manifest.
"""

import json
import re
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------

DRIVE_ROOT = Path("/content/drive/MyDrive/AI_Agri_GraphRAG")
EVAL_ROOT = DRIVE_ROOT / "evaluation" / "model_eval_dev_v2"
MANIFEST_PATH = DRIVE_ROOT / "evaluation" / "manifests" / "agmmu_retrieval_dev_v1.json"

METHODS = {
    "No-RAG": EVAL_ROOT / "no_rag" / "predictions.jsonl",
    "VectorRAG": EVAL_ROOT / "vector_rag" / "predictions.jsonl",
    "GraphRAG": EVAL_ROOT / "graph_rag" / "predictions.jsonl",
}

EXPECTED_CASES = 30

# ---------------------------------------------------------------------
# 2. FILE HELPERS
# ---------------------------------------------------------------------

def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
    return rows


def get_first(record, keys):
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def get_id(record):
    value = get_first(record, ["faq_id", "faq-id", "faq_id", "id", "example_id"])
    return str(value).strip() if value is not None else ""


def get_ground_truth(record):
    return get_first(
        record,
        [
            "ground_truth",
            "ground_truth_answer",
            "reference_answer",
            "expected_answer",
            "label",
        ],
    )


def get_prediction(record):
    return get_first(
        record,
        [
            "prediction",
            "predicted_answer",
            "model_answer",
            "generated_answer",
            "response",
            "output",
        ],
    )


# ---------------------------------------------------------------------
# 3. LOAD THE DEVELOPMENT MANIFEST (OPTIONS ONLY; NOT USED TO INFER GT)
# ---------------------------------------------------------------------

manifest_data = load_json(MANIFEST_PATH)

if isinstance(manifest_data, dict):
    manifest_records = manifest_data.get(
        "records", manifest_data.get("data", [])
    )
else:
    manifest_records = manifest_data

if not isinstance(manifest_records, list):
    raise TypeError("Development manifest must be a list or contain records/data.")

manifest_by_id = {}
for record in manifest_records:
    faq_id = get_id(record)
    if not faq_id:
        raise ValueError(f"Manifest record is missing an FAQ ID: {record}")
    if faq_id in manifest_by_id:
        raise ValueError(f"Duplicate FAQ ID in manifest: {faq_id}")
    manifest_by_id[faq_id] = record

if len(manifest_by_id) != EXPECTED_CASES:
    raise ValueError(
        f"Expected {EXPECTED_CASES} development manifest records, "
        f"found {len(manifest_by_id)}."
    )


# ---------------------------------------------------------------------
# 4. ANSWER EXTRACTION
# ---------------------------------------------------------------------

def extract_answer(raw):
    """Extract the answer without explanation text; empty outputs stay empty."""
    if raw is None:
        return ""

    text = str(raw).strip()
    if not text:
        return ""

    # Remove optional code fences.
    text = re.sub(
        r"^\s*```(?:text|json|markdown)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    split_pattern = (
        r"</answer\s*>"
        r"|<brief explanation\s*>"
        r"|<explanation\s*>"
        r"|^\s*(?:Explanation|Brief explanation)\s*:"
        r"|\n[ \t]*\n"
    )

    # Format: <answer> ... </answer>, including malformed missing closing tags.
    tag = re.search(r"<answer\s*>", text, flags=re.IGNORECASE)
    if tag:
        text = text[tag.end():]
        text = re.split(
            split_pattern,
            text,
            maxsplit=1,
            flags=re.IGNORECASE | re.MULTILINE,
        )[0]
    else:
        # Format: Answer: ...
        header = re.search(
            r"^\s*Answer\s*:\s*",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not header:
            return ""

        text = text[header.end():]

        first_line = text.splitlines()[0] if text.splitlines() else ""
        if re.fullmatch(
            r"\s*<answer\s*/?>\s*",
            first_line,
            flags=re.IGNORECASE,
        ):
            return ""

        text = re.split(
            split_pattern,
            text,
            maxsplit=1,
            flags=re.IGNORECASE | re.MULTILINE,
        )[0]

    text = text.strip().strip("\"'` ")
    text = re.sub(r"</?answer\s*>", "", text, flags=re.IGNORECASE).strip()

    if not text or re.match(
        r"^(?:Explanation|Brief explanation)\s*:",
        text,
        flags=re.IGNORECASE,
    ):
        return ""

    return text


# ---------------------------------------------------------------------
# 5. OPTION NUMBER RESOLUTION AND NORMALIZATION
# ---------------------------------------------------------------------

def parse_options(options):
    if options is None:
        return []

    if isinstance(options, list):
        return [str(option).strip() for option in options]

    if isinstance(options, str):
        raw = options.strip()
        if not raw:
            return []

        # Some records may serialize options as JSON.
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, list):
                return [str(option).strip() for option in decoded]
        except (json.JSONDecodeError, TypeError):
            pass

        # Otherwise treat each non-empty line as an option.
        return [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            for line in raw.splitlines()
            if line.strip()
        ]

    return []


def resolve_option_number(answer, options):
    """
    If the entire answer is only a choice number (e.g. '4' or '4.'),
    resolve it to the option text from the matching manifest record.
    If answer already includes option text (e.g. '4. leaf spot'), leave it
    intact; normalization will remove its leading number.
    """
    answer = str(answer or "").strip()
    options = parse_options(options)

    match = re.fullmatch(r"(?:option\s*)?(\d+)[.)]?", answer, flags=re.IGNORECASE)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(options):
            return options[index]

    return answer


def normalize_answer(answer):
    """Normalize case, option prefixes, whitespace, and terminal punctuation."""
    if answer is None:
        return ""

    text = str(answer).strip().lower()

    # Remove prefixes such as "Answer:" if any remain.
    text = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.IGNORECASE)

    # Remove numbered option prefix only when followed by answer text.
    text = re.sub(r"^\s*\d+\s*[.)]\s*", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \"'`")
    text = text.rstrip(" \t\r\n.,;:!?")
    return text.strip()


# ---------------------------------------------------------------------
# 6. SCORE EACH METHOD
# ---------------------------------------------------------------------

all_audits = {}
summary = []
id_sets = {}

for method_name, prediction_path in METHODS.items():
    print("\n" + "=" * 70)
    print(f"SCORING: {method_name}")
    print("=" * 70)

    rows = load_jsonl(prediction_path)

    # Ignore explicitly failed rows, but require all 30 successful predictions.
    rows = [
        row for row in rows
        if str(row.get("status", "success")).lower()
        in ("success", "ok", "completed")
    ]

    by_id = {}
    for row in rows:
        faq_id = get_id(row)
        if not faq_id:
            raise ValueError(f"{method_name}: prediction row is missing FAQ ID.")
        if faq_id in by_id:
            raise ValueError(
                f"{method_name}: duplicate prediction for FAQ ID {faq_id}. "
                "Do not silently score duplicate outputs."
            )
        by_id[faq_id] = row

    prediction_ids = set(by_id)
    manifest_ids = set(manifest_by_id)

    missing_ids = sorted(manifest_ids - prediction_ids)
    unexpected_ids = sorted(prediction_ids - manifest_ids)

    if missing_ids or unexpected_ids:
        raise ValueError(
            f"{method_name}: prediction IDs do not match the development "
            f"manifest. Missing: {missing_ids}; unexpected: {unexpected_ids}"
        )

    results = []

    # Score in manifest order for consistent, auditable output.
    for faq_id, manifest_record in manifest_by_id.items():
        row = by_id[faq_id]

        ground_truth = get_ground_truth(row)
        if ground_truth is None:
            # Ground truth can be read from the manifest if absent in prediction.
            ground_truth = get_first(
                manifest_record,
                ["answer", "ground_truth", "ground_truth_answer", "label"],
            )

        if ground_truth is None:
            raise KeyError(
                f"{method_name}, FAQ {faq_id}: ground truth not found "
                "in prediction or manifest."
            )

        raw_prediction = get_prediction(row)
        extracted = extract_answer(raw_prediction)

        # Use only the options belonging to this example.
        options = manifest_record.get(
            "options", row.get("options", [])
        )
        resolved = resolve_option_number(extracted, options)

        normalized_gt = normalize_answer(ground_truth)
        normalized_pred = normalize_answer(resolved)

        correct = bool(normalized_gt) and bool(normalized_pred) and (
            normalized_gt == normalized_pred
        )

        results.append({
            "faq_id": faq_id,
            "ground_truth": str(ground_truth),
            "raw_prediction": str(raw_prediction or ""),
            "extracted_answer": extracted,
            "resolved_answer": resolved,
            "normalized_ground_truth": normalized_gt,
            "normalized_prediction": normalized_pred,
            "correct": bool(correct),
            "status": row.get("status", "success"),
        })

    total = len(results)
    correct_count = sum(item["correct"] for item in results)
    incorrect_count = total - correct_count
    accuracy = (100.0 * correct_count / total) if total else 0.0

    output_dir = prediction_path.parent
    audit_path = output_dir / "shared_scoring_audit_standardized.json"

    audit = {
        "scoring_version": "standardized_v1",
        "method": method_name,
        "dataset": "agmmu_retrieval_dev_v1 (30-case development set)",
        "metric": "normalized exact match",
        "option_number_policy": (
            "A bare option number is resolved using the options for that FAQ "
            "in the development manifest."
        ),
        "cases_scored": total,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "accuracy_percent": round(accuracy, 2),
        "results": results,
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    all_audits[method_name] = audit
    id_sets[method_name] = prediction_ids

    summary.append({
        "method": method_name,
        "cases": total,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "accuracy_percent": round(accuracy, 2),
        "audit_path": str(audit_path),
    })

    print(f"Cases scored: {total}")
    print(f"Correct:      {correct_count}/{total}")
    print(f"Incorrect:    {incorrect_count}/{total}")
    print(f"Accuracy:     {accuracy:.2f}%")
    print(f"Saved audit:  {audit_path}")
    print("\nIncorrect cases:")

    for item in results:
        if not item["correct"]:
            print(
                f"\nFAQ {item['faq_id']}"
                f"\n  GT:   {item['ground_truth']}"
                f"\n  Pred: {item['resolved_answer'][:300]}"
            )

    if total != EXPECTED_CASES:
        raise ValueError(
            f"{method_name}: expected {EXPECTED_CASES} scored cases, got {total}."
        )


# ---------------------------------------------------------------------
# 7. SAVE CONSOLIDATED SUMMARY
# ---------------------------------------------------------------------

summary_path = EVAL_ROOT / "three_method_dev_summary_standardized.json"
summary_path.parent.mkdir(parents=True, exist_ok=True)

summary_data = {
    "scoring_version": "standardized_v1",
    "dataset": "30-case development set",
    "metric": "normalized exact-match accuracy",
    "methods": summary,
}

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print("THREE-METHOD STANDARDIZED SUMMARY")
print("=" * 70)

for item in summary:
    print(
        f"{item['method']:12s} | "
        f"{item['correct']:2d}/{item['cases']:2d} | "
        f"{item['accuracy_percent']:.2f}%"
    )

print(f"\nSaved consolidated summary:\n{summary_path}")


# ---------------------------------------------------------------------
# 8. CROSS-METHOD ID CONSISTENCY
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("CROSS-METHOD ID CONSISTENCY")
print("=" * 70)

for method_name, ids in id_sets.items():
    print(f"{method_name}: {len(ids)} unique FAQ IDs")

same_ids = (
    len(id_sets) == 3
    and len({frozenset(ids) for ids in id_sets.values()}) == 1
    and next(iter(id_sets.values())) == set(manifest_by_id)
)

print(f"Same FAQ ID set across all methods: {same_ids}")

if not same_ids:
    for method_name, ids in id_sets.items():
        print(f"{method_name} missing: {sorted(set(manifest_by_id) - ids)}")
        print(f"{method_name} unexpected: {sorted(ids - set(manifest_by_id))}")

print("\nScoring complete.")
