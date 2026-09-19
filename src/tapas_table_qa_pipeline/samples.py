"""Labelled table-question datasets for the adaptation contract: the digest-pinned WikiSQL sample, the record
contract and its structural validation, table-disjoint splitting, and the BYOD JSONL loader.

A record is ``{id, table, question, answer}`` where ``table`` is ``{column: [cells]}`` (every header and cell a
str), ``question`` a non-empty str and ``answer`` the supervision in the pipeline's own vocabulary:
``{"aggregation": NONE|SUM|AVERAGE|COUNT, "coordinates": [[row, col], ...], "denotation": [cell, ...] | number}``.
For ``NONE`` the denotation is the selected cells; for the three operators it is the number the operator
should produce from the selected cells (the pipeline computes it the same way at inference). An optional
``category`` (free text, at most 32 characters) is carried into the per-category breakdown.

The default sample is drawn from the WikiSQL validation shard (BSD-3-Clause, Zhong et al. 2017) converted to
parquet by the Hugging Face Hub at an immutable revision: the shard is fetched whole (3.6 MB), refused on any
byte-count or SHA-256 mismatch, its SQL programme executed in pure Python to obtain the gold cells and value,
and a seeded, table-disjoint, aggregation-stratified draw taken. WikiSQL's MAX/MIN become ``NONE`` with the
extreme cell selected (TAPAS-WTQ has no MAX/MIN operator); its AVG is the pipeline's ``AVERAGE``. The original
operator is kept as ``category`` (``lookup``, ``max``, ``min``, ``count``, ``sum``, ``average``).
"""
# ruff: noqa: E501  -- record and pin literals are kept on single lines

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import (
    AGGREGATIONS,
    MODEL_ID,
    _check_query,
    _check_table,
    compute_numeric_answer,
    parse_number,
)

CORPUS_NAME = "WikiSQL (validation shard)"
CORPUS_REPO = "Salesforce/wikisql"
CORPUS_REVISION = "48cfb60afd0d5f9d2231ca90f76edf9f975181bc"  # refs/convert/parquet commit on the Hub
CORPUS_FILE = "default/validation/0000.parquet"
CORPUS_SHA256 = "ed524cc7221e6c0010c7b57691da40897b38dce65090ecff48836989d606a347"
CORPUS_BYTES = 3_630_670
CORPUS_ROWS = 8_421
CORPUS_LICENSE = "BSD-3-Clause (Zhong, Xiong & Socher 2017, https://github.com/salesforce/WikiSQL)"
CORPUS_URL = f"https://huggingface.co/datasets/{CORPUS_REPO}/resolve/{CORPUS_REVISION}/{CORPUS_FILE}"
DEFAULT_CACHE_DIR = Path("weights") / "wikisql"

# Candidate filter (the sample must fit MAX_TOKENS untruncated so every gold cell is visible to the model).
SAMPLE_MAX_ROWS = 20
SAMPLE_MAX_COLUMNS = 12
SAMPLE_MAX_CHARS = 1_000  # header + cells
SAMPLE_MAX_LOOKUP_CELLS = 4
SAMPLE_SEED = 42
# Per WikiSQL operator. Training keeps lookups the majority (80 of 240) as they are in WikiSQL and in most table-QA
# use, so the adaptation does not trade lookup accuracy for operator accuracy; validation and test are stratified
# evenly so every operator is read on the same footing (90 / 150).
SAMPLE_SPLIT: dict[str, int | dict[str, int]] = {
    "train": {"lookup": 80, "count": 32, "sum": 32, "average": 32, "max": 32, "min": 32},
    "validation": 15,
    "test": 25,
}
SAMPLE_TABLE_FRACTIONS = {"train": 0.55, "validation": 0.2}  # of the candidate tables; the rest (0.25) is test
WIKISQL_AGGREGATIONS = ("NONE", "MAX", "MIN", "COUNT", "SUM", "AVG")
CATEGORY_OF = {"NONE": "lookup", "MAX": "max", "MIN": "min", "COUNT": "count", "SUM": "sum", "AVG": "average"}
SAMPLE_DIGEST = "806362262fe311efd659049821f857946ad7f4188385172f9c3d0411e919463e"  # dataset_digest over the three default splits together; tests pin it

MIN_RECORDS = 8
MAX_RECORDS = 20_000
MAX_CATEGORY_CHARS = 32
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


# ---------------------------------------------------------------------------------------------------------
# WikiSQL: fetch, execute, draw
# ---------------------------------------------------------------------------------------------------------


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> bytes:
    """Return the pinned WikiSQL validation shard from the cache or the Hub; refuse any size/digest mismatch."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / "wikisql-validation.parquet"
    data = local.read_bytes() if local.is_file() else b""
    if len(data) != CORPUS_BYTES or _sha256_bytes(data) != CORPUS_SHA256:
        if fetcher is not None:
            data = fetcher(CORPUS_URL)
        else:
            request = urllib.request.Request(CORPUS_URL, headers={"User-Agent": "tapas-table-qa-pipeline"})
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - pinned https URL
                data = response.read()
        if len(data) != CORPUS_BYTES:
            raise ValueError(f"{CORPUS_FILE}: {len(data)} bytes, pinned {CORPUS_BYTES}")
        digest = _sha256_bytes(data)
        if digest != CORPUS_SHA256:
            raise ValueError(f"{CORPUS_FILE}: sha256 {digest} != pinned {CORPUS_SHA256}")
        local.write_bytes(data)
    return data


def read_corpus(data: bytes) -> list[dict[str, Any]]:
    """Decode the shard's ``question``, ``table`` and ``sql`` columns into plain dicts (index = shard row)."""
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(data), columns=["question", "table", "sql"])
    if table.num_rows != CORPUS_ROWS:
        raise ValueError(f"{CORPUS_FILE}: {table.num_rows} rows, pinned {CORPUS_ROWS}")
    out = []
    for index, row in enumerate(table.to_pylist()):
        row["index"] = index
        out.append(row)
    return out


def _matching_rows(rows: Sequence[Sequence[str]], conds: Mapping[str, Sequence[Any]]) -> list[int]:
    keep = []
    for i, row in enumerate(rows):
        ok = True
        for col, op, cond in zip(conds["column_index"], conds["operator_index"], conds["condition"], strict=True):
            cell = row[col]
            if op == 0:
                ok = _normalise(cell) == _normalise(cond)
            else:
                a, b = parse_number(str(cell)), parse_number(str(cond))
                ok = a is not None and b is not None and (a > b if op == 1 else a < b)
            if not ok:
                break
        if ok:
            keep.append(i)
    return keep


def execute_sql(example: Mapping[str, Any]) -> dict[str, Any] | None:
    """Run one WikiSQL programme on its table. Returns the record's ``answer`` in the pipeline's vocabulary plus
    ``category``, or None when the programme has no usable denotation (no matching row, unparseable numbers)."""
    table, sql = example["table"], example["sql"]
    rows, sel, op = table["rows"], int(sql["sel"]), WIKISQL_AGGREGATIONS[int(sql["agg"])]
    matched = _matching_rows(rows, sql["conds"])
    if not matched:
        return None
    cells = [str(rows[r][sel]) for r in matched]
    coords = [[r, sel] for r in matched]
    category = CATEGORY_OF[op]
    if op == "NONE":
        return {"aggregation": "NONE", "coordinates": coords, "denotation": cells, "category": category}
    if op == "COUNT":
        return {"aggregation": "COUNT", "coordinates": coords, "denotation": float(len(cells)), "category": category}
    parsed = [parse_number(c) for c in cells]
    if any(v is None for v in parsed):
        return None
    if op in ("MAX", "MIN"):
        pick = max(range(len(parsed)), key=lambda i: parsed[i]) if op == "MAX" else min(range(len(parsed)), key=lambda i: parsed[i])
        return {"aggregation": "NONE", "coordinates": [coords[pick]], "denotation": [cells[pick]], "category": category}
    aggregation = "SUM" if op == "SUM" else "AVERAGE"
    value, _ = compute_numeric_answer(cells, aggregation)
    return {"aggregation": aggregation, "coordinates": coords, "denotation": float(value), "category": category}


def table_digest(table: Mapping[str, Sequence[str]]) -> str:
    """SHA-256 of the normalised header and cells: the identity a split is made disjoint on."""
    columns, rows = _check_table(table)
    payload = json.dumps([[_normalise(c) for c in columns], [[_normalise(v) for v in row] for row in rows]])
    return _sha256_bytes(payload.encode("utf-8"))


def wikisql_candidates(examples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Records for every shard example whose table fits the sample ceilings and whose programme executes."""
    out = []
    for ex in examples:
        tb = ex["table"]
        header, rows = [str(h) for h in tb["header"]], [[str(c) for c in row] for row in tb["rows"]]
        if not 1 <= len(rows) <= SAMPLE_MAX_ROWS or not 1 <= len(header) <= SAMPLE_MAX_COLUMNS:
            continue
        if any(len(c) > 200 for row in rows for c in row) or any(len(h) > 200 for h in header):
            continue
        if sum(len(c) for row in rows for c in row) + sum(len(h) for h in header) > SAMPLE_MAX_CHARS:
            continue
        if len(set(header)) != len(header):
            continue
        answer = execute_sql({"table": {"rows": rows}, "sql": ex["sql"]})
        if answer is None or (answer["aggregation"] == "NONE" and len(answer["coordinates"]) > SAMPLE_MAX_LOOKUP_CELLS):
            continue
        category = answer.pop("category")
        out.append(
            {
                "id": f"wikisql-val-{ex['index']}",
                "table": {h: [row[i] for row in rows] for i, h in enumerate(header)},
                "question": str(ex["question"]),
                "answer": answer,
                "category": category,
                "source_table_id": str(tb["id"]),
                "sql": str(ex["sql"]["human_readable"]),
            }
        )
    return out


def build_sample_dataset(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int | Mapping[str, int]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded draw: tables are shuffled and assigned to a split first (so no table crosses splits), then
    ``sizes[split]`` records per WikiSQL operator (one int for all, or a per-category mapping) are drawn from
    each split's pool."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    tables = sorted({str(c["source_table_id"]) for c in candidates})
    rng.shuffle(tables)
    n = len(tables)
    train_end = int(n * SAMPLE_TABLE_FRACTIONS["train"])
    val_end = train_end + int(n * SAMPLE_TABLE_FRACTIONS["validation"])
    split_of = {t: ("train" if i < train_end else "validation" if i < val_end else "test") for i, t in enumerate(tables)}
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {name: {} for name in sizes}
    for c in candidates:
        pools[split_of[str(c["source_table_id"])]].setdefault(str(c["category"]), []).append(dict(c))
    out: dict[str, list[dict[str, Any]]] = {}
    for name, spec in sizes.items():
        drawn = []
        for category in sorted(pools[name]):
            per = int(spec[category]) if isinstance(spec, Mapping) else int(spec)
            items = pools[name][category]
            rng.shuffle(items)
            if len(items) < per:
                raise ValueError(f"{name}/{category}: only {len(items)} candidates, need {per}")
            drawn.extend(items[:per])
        rng.shuffle(drawn)
        out[name] = drawn
    return out


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, seed: int = SAMPLE_SEED) -> dict[str, list[dict[str, Any]]]:
    """fetch → read → candidates → draw, in one call."""
    return build_sample_dataset(wikisql_candidates(read_corpus(fetch_corpus(cache_dir=cache_dir))), seed=seed)


# ---------------------------------------------------------------------------------------------------------
# Record contract
# ---------------------------------------------------------------------------------------------------------


def _check_answer(answer: Any, columns: Sequence[str], rows: Sequence[Sequence[str]], where: str) -> dict[str, Any]:
    if not isinstance(answer, Mapping):
        raise ValueError(f"{where}: answer must be a mapping with aggregation/coordinates/denotation")
    aggregation = answer.get("aggregation")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"{where}: answer.aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    coords = answer.get("coordinates")
    if not isinstance(coords, Sequence) or isinstance(coords, str | bytes) or not coords:
        raise ValueError(f"{where}: answer.coordinates must be a non-empty list of [row, col] pairs")
    checked_coords = []
    for pair in coords:
        if not isinstance(pair, Sequence) or isinstance(pair, str | bytes) or len(pair) != 2:
            raise ValueError(f"{where}: answer.coordinates entries must be [row, col] pairs")
        r, c = pair
        if isinstance(r, bool) or isinstance(c, bool) or not isinstance(r, int) or not isinstance(c, int):
            raise ValueError(f"{where}: answer.coordinates must be integer pairs")
        if not (0 <= r < len(rows) and 0 <= c < len(columns)):
            raise ValueError(f"{where}: coordinate {[r, c]} outside the {len(rows)}x{len(columns)} table")
        if [r, c] in checked_coords:
            raise ValueError(f"{where}: duplicate coordinate {[r, c]}")
        checked_coords.append([r, c])
    denotation = answer.get("denotation")
    cells = [rows[r][c] for r, c in checked_coords]
    if aggregation == "NONE":
        if not isinstance(denotation, Sequence) or isinstance(denotation, str | bytes) or not all(isinstance(d, str) for d in denotation):
            raise ValueError(f"{where}: a NONE answer's denotation must be a list of cell strings")
        if sorted(_normalise(d) for d in denotation) != sorted(_normalise(c) for c in cells):
            raise ValueError(f"{where}: NONE denotation {list(denotation)} does not match the selected cells {cells}")
        checked_denotation: Any = [str(d) for d in denotation]
    else:
        if isinstance(denotation, bool) or not isinstance(denotation, int | float):
            raise ValueError(f"{where}: a {aggregation} answer's denotation must be a number")
        value, unparsed = compute_numeric_answer(cells, aggregation)
        if value is None:
            raise ValueError(f"{where}: {aggregation} over cells that do not parse as numbers: {unparsed}")
        if abs(float(value) - float(denotation)) > 1e-6:
            raise ValueError(f"{where}: {aggregation} of the selected cells is {value}, denotation says {denotation}")
        checked_denotation = float(denotation)
    return {"aggregation": aggregation, "coordinates": checked_coords, "denotation": checked_denotation}


def _check_record(record: Any, index: int) -> dict[str, Any]:
    where = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{where} must be a mapping with id/table/question/answer")
    for key in ("id", "table", "question", "answer"):
        if key not in record:
            raise ValueError(f"{where} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{where}: id must match {_ID_RE.pattern}")
    columns, rows = _check_table(record["table"])
    question = _check_query(record["question"])
    answer = _check_answer(record["answer"], columns, rows, where)
    item = {"id": rid, "table": {h: [row[i] for row in rows] for i, h in enumerate(columns)}, "question": question, "answer": answer}
    category = record.get("category")
    if category is not None:
        if not isinstance(category, str) or not 1 <= len(category.strip()) <= MAX_CATEGORY_CHARS:
            raise ValueError(f"{where}: category must be a str of 1..{MAX_CATEGORY_CHARS} characters")
        item["category"] = category.strip()
    for key in ("source_table_id", "sql"):
        if key in record:
            item[key] = str(record[key])
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
) -> dict[str, Any]:
    """Structural validation of a labelled table-question dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, table, question, answer} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked, ids, aggregations, categories, tables = [], set(), {}, {}, set()
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        aggregations[item["answer"]["aggregation"]] = aggregations.get(item["answer"]["aggregation"], 0) + 1
        if "category" in item:
            categories[item["category"]] = categories.get(item["category"], 0) + 1
        tables.add(table_digest(item["table"]))
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "n_tables": len(tables),
        "aggregation_counts": dict(sorted(aggregations.items())),
        "category_counts": dict(sorted(categories.items())),
        "table_shape": {
            "rows": [min(len(next(iter(r["table"].values()))) for r in checked), max(len(next(iter(r["table"].values()))) for r in checked)],
            "columns": [min(len(r["table"]) for r in checked), max(len(r["table"]) for r in checked)],
        },
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent SHA-256 over (id, table digest, question, answer)."""
    parts = sorted(
        json.dumps([r["id"], table_digest(r["table"]), r["question"], r["answer"]], sort_keys=True) for r in records
    )
    return _sha256_bytes("\n".join(parts).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no table (by normalised content) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = table_digest(record["table"])
            if key in seen and seen[key] != name:
                raise ValueError(f"table of {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded table-disjoint split of a BYOD dataset: tables are shuffled and cut by fraction, and every record of
    a table follows its table."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    by_table: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        by_table.setdefault(table_digest(record["table"]), []).append(record)
    keys = sorted(by_table)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_test = max(1, round(n * test_fraction))
    n_val = round(n * val_fraction)
    if n - n_test - n_val < 1:
        raise ValueError(f"{n} distinct tables are too few to split into train/validation/test")
    out = {"train": [], "validation": [], "test": []}
    for i, key in enumerate(keys):
        name = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
        out[name].extend(by_table[key])
    return out


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read records from a JSONL file (one record per line) or a JSON list; validation happens downstream."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)  # one JSON list of records
        except json.JSONDecodeError:
            data = None  # not a single document: read it as JSONL below
        if isinstance(data, list):
            if any(not isinstance(r, Mapping) for r in data):
                raise ValueError("each JSONL line must be a record object")
            return data
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if any(not isinstance(r, Mapping) for r in rows):
        raise ValueError("each JSONL line must be a record object")
    return rows


def write_dataset_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write records as JSONL in the BYOD shape."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for record in records:
            row = {k: record[k] for k in ("id", "table", "question", "answer") if k in record}
            if "category" in record:
                row["category"] = record["category"]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """A human-readable summary (id, category, aggregation, question, sql, denotation), not the BYOD format."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "category", "aggregation", "question", "sql", "denotation"])
        for r in records:
            d = r["answer"]["denotation"]
            writer.writerow([r["id"], r.get("category", ""), r["answer"]["aggregation"], r["question"], r.get("sql", ""), json.dumps(d) if isinstance(d, list) else d])
    return out
