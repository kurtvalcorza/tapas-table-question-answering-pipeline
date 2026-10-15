"""Labelled table-question datasets for the adaptation contract: the digest-pinned financial-table sample, the
record contract and its structural validation, table-disjoint splitting, and the BYOD JSONL loader.

A record is ``{id, table, question, answer}`` where ``table`` is ``{column: [cells]}`` (every header and cell a
str), ``question`` a non-empty str and ``answer`` the supervision in the pipeline's own vocabulary:
``{"aggregation": NONE|SUM|AVERAGE|COUNT, "coordinates": [[row, col], ...], "denotation": [cell, ...] | number}``.
For ``NONE`` the denotation is the selected cells; for the three operators it is the number the operator
should produce from the selected cells (the pipeline computes it the same way at inference). An optional
``category`` (free text, at most 32 characters) is carried into the per-category breakdown.

The default sample comes from two table-QA corpora that are **not** in the pinned checkpoint's fine-tuning chain
(SQA → WikiSQL → WTQ, all over Wikipedia tables). Both are built from company financial reports:

- **TAT-QA** (Zhu et al. 2021, CC BY 4.0) supplies the ``lookup`` questions: table-answered span questions whose
  answer cells occur exactly once in the table (``NONE`` over those cells).
- **FinQA** (Chen et al. 2021, MIT on GitHub; CC BY 4.0 on its Hub card) supplies the ``sum`` and ``average`` questions: table-only
  programmes that are a chain of ``add`` (``SUM``), the same chain divided by its number of terms (``AVERAGE``),
  or ``table_sum`` / ``table_average`` over one row. Each term must match exactly one table cell.

Neither corpus has questions this conversion can map to ``COUNT``, so **COUNT is not exercised by the default
sample** (the operator stays in the pipeline and in BYOD). Each file is fetched whole at an immutable commit and
refused on any byte-count or SHA-256 mismatch. Cells are normalised so the pipeline's number parser reads them
(``$ 1,234`` → ``1,234``, ``( 12 )`` → ``-12``, ``$ -23158 ( 23158 )`` → ``-23158``). The draw is seeded and
group-disjoint: a FinQA report (one company's filing for one year), or a TAT-QA table, never spans two
splits; FinQA's own train, dev and test files are pooled before the draw.
"""
# ruff: noqa: E501  -- record and pin literals are kept on single lines

from __future__ import annotations

import csv
import hashlib
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

# Pinned sources. The FinQA files are served from GitHub at a commit (immutable); the TAT-QA file from the Hub at a
# dataset revision (immutable). FinQA's GitHub licence is MIT; the Hub card of ibm-research/finqa says CC BY 4.0, so
# the notebook follows the stricter reading and attributes it as CC BY 4.0 as well.
CORPORA: dict[str, dict[str, Any]] = {
    "finqa-train": {
        "name": "FinQA (train split)",
        "url": "https://raw.githubusercontent.com/czyssrs/FinQA/0f16e2867befa6840783e58be38c9efb9229d742/dataset/train.json",
        "revision": "0f16e2867befa6840783e58be38c9efb9229d742",
        "bytes": 78_216_616,
        "sha256": "49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6",
        "license": "MIT (github.com/czyssrs/FinQA); the Hub card ibm-research/finqa says CC BY 4.0, so attribution is kept",
        "citation": "Z. Chen et al., FinQA: A Dataset of Numerical Reasoning over Financial Data, EMNLP 2021",
    },
    "finqa-dev": {
        "name": "FinQA (dev split)",
        "url": "https://raw.githubusercontent.com/czyssrs/FinQA/0f16e2867befa6840783e58be38c9efb9229d742/dataset/dev.json",
        "revision": "0f16e2867befa6840783e58be38c9efb9229d742",
        "bytes": 10_954_658,
        "sha256": "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
        "license": "MIT (github.com/czyssrs/FinQA); the Hub card ibm-research/finqa says CC BY 4.0, so attribution is kept",
        "citation": "Z. Chen et al., FinQA: A Dataset of Numerical Reasoning over Financial Data, EMNLP 2021",
    },
    "finqa-test": {
        "name": "FinQA (test split)",
        "url": "https://raw.githubusercontent.com/czyssrs/FinQA/0f16e2867befa6840783e58be38c9efb9229d742/dataset/test.json",
        "revision": "0f16e2867befa6840783e58be38c9efb9229d742",
        "bytes": 14_395_143,
        "sha256": "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
        "license": "MIT (github.com/czyssrs/FinQA); the Hub card ibm-research/finqa says CC BY 4.0, so attribution is kept",
        "citation": "Z. Chen et al., FinQA: A Dataset of Numerical Reasoning over Financial Data, EMNLP 2021",
    },
    "tatqa-train": {
        "name": "TAT-QA (train split)",
        "url": "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/c96247f5077eac447f63527fd3dcfdc58bb56d6a/tatqa_dataset_train.json",
        "revision": "c96247f5077eac447f63527fd3dcfdc58bb56d6a",
        "bytes": 12_827_783,
        "sha256": "c7bf8eb0c757e57ef3db6c6146dd7e487100dac1e37e66a349813ddc4187cd65",
        "license": "CC BY 4.0 (next-tat/TAT-QA dataset card; github.com/NExTplusplus/TAT-QA README)",
        "citation": "F. Zhu et al., TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance, ACL 2021",
    },
}
CORPUS_NAME = "TAT-QA lookups + FinQA sums and averages (financial-report tables)"
CORPUS_LICENSE = "TAT-QA: CC BY 4.0; FinQA: MIT (GitHub), CC BY 4.0 on its Hub card — attribution kept for both"
CORPUS_PINS = {key: {k: spec[k] for k in ("url", "bytes", "sha256")} for key, spec in CORPORA.items()}
DEFAULT_CACHE_DIR = Path("weights") / "table-qa-corpora"

# Candidate filter (the sample must fit MAX_TOKENS untruncated so every gold cell is visible to the model).
SAMPLE_MAX_ROWS = 20
SAMPLE_MAX_COLUMNS = 12
SAMPLE_MAX_CHARS = 1_000  # header + cells
SAMPLE_MAX_LOOKUP_CELLS = 4
SAMPLE_SEED = 42
# Per category. COUNT has no source in this revision and is not drawn. Training keeps lookups the largest share,
# so the adaptation does not trade lookup accuracy for operator accuracy; validation and test are stratified
# evenly so every category is read on the same footing.
SAMPLE_SPLIT: dict[str, int | dict[str, int]] = {
    "train": {"lookup": 80, "sum": 60, "average": 60},
    "validation": 20,
    "test": 35,
}
SAMPLE_CATEGORIES = ("lookup", "sum", "average")
SAMPLE_TABLE_FRACTIONS = {"train": 0.5, "validation": 0.2}  # of each source's groups; the rest (0.3) is test
SAMPLE_DIGEST = "eda5c8ce8e5c2c62cec4c52d89e89abf49ca574736460e73330ed0cfd30a58d5"  # dataset_digest over the three default splits together; tests pin it

MIN_RECORDS = 8
MAX_RECORDS = 20_000
MAX_CATEGORY_CHARS = 32
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


# ---------------------------------------------------------------------------------------------------------
# Corpora: fetch, convert, draw
# ---------------------------------------------------------------------------------------------------------


def fetch_corpus(key: str, *, cache_dir: str | Path | None = None, fetcher: Any = None) -> bytes:
    """Return one pinned corpus file from the cache or its source; refuse any size/digest mismatch."""
    spec = CORPORA[key]
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{key}.json"
    data = local.read_bytes() if local.is_file() else b""
    if len(data) != spec["bytes"] or _sha256_bytes(data) != spec["sha256"]:
        if fetcher is not None:
            data = fetcher(spec["url"])
        else:
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "tapas-table-qa-pipeline"})
            with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 - pinned https URL
                data = response.read()
        if len(data) != spec["bytes"]:
            raise ValueError(f"{key}: {len(data)} bytes, pinned {spec['bytes']}")
        digest = _sha256_bytes(data)
        if digest != spec["sha256"]:
            raise ValueError(f"{key}: sha256 {digest} != pinned {spec['sha256']}")
        local.write_bytes(data)
    return data


_PAREN_NEG = re.compile(r"^\(\s*\$?\s*(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%?)\s*\)$")
_PLAIN = re.compile(r"^\$?\s*(-?)\s*\$?\s*(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%?)(?:\s*\(\s*[\d,.]+\s*%?\s*\))?$")


def clean_cell(text: Any) -> str:
    """Whitespace-collapse a cell and rewrite currency/negative formats so ``parse_number`` reads them; any other
    text is returned unchanged (apart from whitespace)."""
    t = " ".join(str(text).split())
    m = _PAREN_NEG.match(t)
    if m:
        return f"-{m.group(1)}{m.group(2) or ''}{m.group(3)}"
    m = _PLAIN.match(t)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3) or ''}{m.group(4)}"
    return t


def convert_table(raw: Sequence[Sequence[Any]]) -> tuple[list[str], list[list[str]], list[list[str]]] | None:
    """First row → header (blank names filled, duplicates suffixed); body cells cleaned. Returns (header, cleaned
    rows, original rows) or None when the table breaks a sample ceiling or is ragged."""
    if len(raw) < 2:
        return None
    width = len(raw[0])
    if any(len(r) != width for r in raw) or not 1 <= width <= SAMPLE_MAX_COLUMNS or not 1 <= len(raw) - 1 <= SAMPLE_MAX_ROWS:
        return None
    header: list[str] = []
    for i, h in enumerate(raw[0]):
        name = " ".join(str(h).split()) or ("item" if i == 0 else f"column {i + 1}")
        while name in header:
            name = f"{name} ({i + 1})"
        header.append(name)
    original = [[" ".join(str(c).split()) for c in row] for row in raw[1:]]
    rows = [[clean_cell(c) for c in row] for row in original]
    if any(len(c) > 200 for row in rows for c in row) or any(len(h) > 200 for h in header):
        return None
    if sum(len(c) for row in rows for c in row) + sum(len(h) for h in header) > SAMPLE_MAX_CHARS:
        return None
    return header, rows, original


def _unique_cell(rows: Sequence[Sequence[str]], want: str) -> list[int] | None:
    hits = [[r, c] for r, row in enumerate(rows) for c, cell in enumerate(row) if cell == want]
    return hits[0] if len(hits) == 1 else None


def tatqa_candidates(contexts: Sequence[Mapping[str, Any]], stats: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """``lookup`` records: table-answered span / multi-span questions whose every answer string occurs exactly
    once among the body cells (at most ``SAMPLE_MAX_LOOKUP_CELLS``). ``stats`` (if given) receives the count of
    eligible questions before and after the table filter."""
    out = []
    stats = stats if stats is not None else {}
    for ctx in contexts:
        eligible = [q for q in ctx["questions"] if q.get("answer_from") == "table" and q.get("answer_type") in ("span", "multi-span")]
        stats["tatqa_table_span_questions"] = stats.get("tatqa_table_span_questions", 0) + len(eligible)
        converted = convert_table(ctx["table"]["table"])
        if converted is None:
            continue
        stats["tatqa_after_table_filter"] = stats.get("tatqa_after_table_filter", 0) + len(eligible)
        header, rows, original = converted
        for q in eligible:
            answers = q["answer"] if isinstance(q["answer"], list) else [q["answer"]]
            answers = [" ".join(str(a).split()) for a in answers]
            if not 1 <= len(answers) <= SAMPLE_MAX_LOOKUP_CELLS:
                continue
            coords = [_unique_cell(original, a) for a in answers]
            if any(c is None for c in coords) or len({tuple(c) for c in coords}) != len(coords):
                continue
            out.append(
                {
                    "id": f"tatqa-{q['uid']}",
                    "table": {h: [row[i] for row in rows] for i, h in enumerate(header)},
                    "question": " ".join(str(q["question"]).split()),
                    "answer": {"aggregation": "NONE", "coordinates": coords, "denotation": [rows[r][c] for r, c in coords]},
                    "category": "lookup",
                    "source": "tatqa",
                    "source_group": f"tatqa:{ctx['table']['uid']}",
                    "source_table_id": str(ctx["table"]["uid"]),
                }
            )
    return out


_STEP = re.compile(r"(\w+)\(([^()]*)\)")


def _finqa_terms(program: str) -> tuple[str, list[str]] | None:
    """(SUM|AVERAGE, leaf terms) for an add-chain, an add-chain divided by its term count, or a one-row
    ``table_sum``/``table_average`` (leaf = ["row:<name>"]); None for any other programme."""
    steps = [(op, [a.strip() for a in args.split(",")]) for op, args in _STEP.findall(program)]
    if len(steps) == 1 and steps[0][0] in ("table_sum", "table_average"):
        return ("SUM" if steps[0][0] == "table_sum" else "AVERAGE"), [f"row:{steps[0][1][0]}"]
    leaves: list[str] = []
    for i, (op, args) in enumerate(steps):
        if len(args) != 2:
            return None
        if op == "add":
            for a in args:
                if a == f"#{i - 1}" and i > 0:
                    continue
                if a.startswith(("#", "const_")):
                    return None
                leaves.append(a)
            if i > 0 and f"#{i - 1}" not in args:
                return None
        elif op == "divide" and i == len(steps) - 1 and i > 0:
            if args != [f"#{i - 1}", f"const_{len(leaves)}"]:
                return None
            return "AVERAGE", leaves
        else:
            return None
    return ("SUM", leaves) if len(leaves) >= 2 else None


def finqa_candidates(examples: Sequence[Mapping[str, Any]], stats: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """``sum`` / ``average`` records from FinQA's table-only programmes (see `_finqa_terms`); every term must match
    exactly one numeric body cell and the pipeline's own arithmetic must reproduce FinQA's executed answer.
    ``stats`` (if given) receives the count of eligible programmes before and after the table filter."""
    out = []
    stats = stats if stats is not None else {}
    for ex in examples:
        qa = ex["qa"]
        if not qa.get("gold_inds") or any(not str(k).startswith("table") for k in qa["gold_inds"]):
            continue
        parsed = _finqa_terms(str(qa.get("program", "")))
        if parsed is None:
            continue
        stats["finqa_table_only_sum_average_programmes"] = stats.get("finqa_table_only_sum_average_programmes", 0) + 1
        converted = convert_table(ex["table"])
        if converted is None:
            continue
        stats["finqa_after_table_filter"] = stats.get("finqa_after_table_filter", 0) + 1
        header, rows, _original = converted
        aggregation, leaves = parsed
        if leaves[0].startswith("row:"):
            name = _normalise(leaves[0][4:])
            matches = [r for r, row in enumerate(rows) if _normalise(row[0]) == name]
            if len(matches) != 1:
                continue
            coords = [[matches[0], c] for c in range(1, len(header))]
            if len(coords) < 2 or any(parse_number(rows[r][c]) is None for r, c in coords):
                continue
        else:
            values = [parse_number(clean_cell(v)) for v in leaves]
            if any(v is None for v in values):
                continue
            coords = []
            for v in values:
                hits = [[r, c] for r, row in enumerate(rows) for c, cell in enumerate(row) if c > 0 and parse_number(cell) is not None and abs(parse_number(cell) - v) < 1e-9]
                if len(hits) != 1:
                    break
                coords.append(hits[0])
            if len(coords) != len(values) or len({tuple(c) for c in coords}) != len(coords):
                continue
        value, unparsed = compute_numeric_answer([rows[r][c] for r, c in coords], aggregation)
        try:
            gold = float(qa["exe_ans"])
        except (TypeError, ValueError):
            continue
        if value is None or unparsed or abs(float(value) - gold) > 1e-3 * max(1.0, abs(gold)):
            continue
        rid = "finqa-" + str(ex["id"]).replace("/", ":")
        if not _ID_RE.match(rid):
            continue
        out.append(
            {
                "id": rid,
                "table": {h: [row[i] for row in rows] for i, h in enumerate(header)},
                "question": " ".join(str(qa["question"]).split()),
                "answer": {"aggregation": aggregation, "coordinates": coords, "denotation": float(value)},
                "category": "sum" if aggregation == "SUM" else "average",
                "source": "finqa",
                "source_group": "finqa:" + "/".join(str(ex["id"]).split("/")[:2]),
                "source_table_id": str(ex["id"]).rsplit("-", 1)[0],
                "program": str(qa["program"]),
            }
        )
    return out


def table_digest(table: Mapping[str, Sequence[str]]) -> str:
    """SHA-256 of the normalised header and cells: the identity a split is made disjoint on."""
    columns, rows = _check_table(table)
    payload = json.dumps([[_normalise(c) for c in columns], [[_normalise(v) for v in row] for row in rows]])
    return _sha256_bytes(payload.encode("utf-8"))


def sample_candidates(corpora: Mapping[str, bytes], stats: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Every candidate record from the pinned files (keys of `CORPORA`), lookups first. ``stats`` (if given)
    receives the filter funnel: eligible questions, after the table filter, and after gold-cell matching."""
    tatqa = json.loads(corpora["tatqa-train"].decode("utf-8"))
    finqa = [ex for key in ("finqa-train", "finqa-dev", "finqa-test") for ex in json.loads(corpora[key].decode("utf-8"))]
    out, seen = [], set()
    stats = stats if stats is not None else {}
    for record in tatqa_candidates(tatqa, stats) + finqa_candidates(finqa, stats):
        if record["id"] not in seen:
            seen.add(record["id"])
            out.append(record)
    stats["tatqa_after_gold_cell_match"] = sum(r["source"] == "tatqa" for r in out)
    stats["finqa_after_gold_cell_match"] = sum(r["source"] == "finqa" for r in out)
    return out


def build_sample_dataset(
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int | Mapping[str, int]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded draw: each source's groups (a FinQA report, i.e. one company's filing for one year; a TAT-QA table) are shuffled and assigned to a split
    first, so no group crosses splits, then ``sizes[split]`` records per category (one int for all, or a
    per-category mapping) are drawn from each split's pool. A table whose normalised content occurs in two groups
    is kept only in the first split it reaches."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    split_of: dict[str, str] = {}
    for source in sorted({str(c.get("source", "")) for c in candidates}):
        groups = sorted({str(c["source_group"]) for c in candidates if str(c.get("source", "")) == source})
        rng.shuffle(groups)
        n = len(groups)
        train_end = int(n * SAMPLE_TABLE_FRACTIONS["train"])
        val_end = train_end + int(n * SAMPLE_TABLE_FRACTIONS["validation"])
        split_of.update({g: ("train" if i < train_end else "validation" if i < val_end else "test") for i, g in enumerate(groups)})
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {name: {} for name in sizes}
    owner: dict[str, str] = {}
    for c in candidates:
        name = split_of[str(c["source_group"])]
        key = table_digest(c["table"])
        if owner.setdefault(key, name) != name:
            continue  # identical table content already assigned to another split
        pools[name].setdefault(str(c["category"]), []).append(dict(c))
    out: dict[str, list[dict[str, Any]]] = {}
    for name, spec in sizes.items():
        drawn = []
        categories = sorted(spec) if isinstance(spec, Mapping) else sorted(pools[name])
        for category in categories:
            per = int(spec[category]) if isinstance(spec, Mapping) else int(spec)
            items = pools[name].get(category, [])
            rng.shuffle(items)
            if len(items) < per:
                raise ValueError(f"{name}/{category}: only {len(items)} candidates, need {per}")
            drawn.extend(items[:per])
        rng.shuffle(drawn)
        out[name] = drawn
    return out


def candidate_summary(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts the notebook prints: questions and distinct groups per category, after conversion and filtering."""
    out: dict[str, Any] = {}
    for category in sorted({str(c["category"]) for c in candidates}):
        items = [c for c in candidates if c["category"] == category]
        out[category] = {"questions": len(items), "groups": len({c["source_group"] for c in items}), "source": items[0].get("source")}
    return out


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, seed: int = SAMPLE_SEED) -> dict[str, list[dict[str, Any]]]:
    """fetch → convert → draw, in one call."""
    corpora = {key: fetch_corpus(key, cache_dir=cache_dir) for key in CORPORA}
    return build_sample_dataset(sample_candidates(corpora), seed=seed)


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
    for key in ("source", "source_group", "source_table_id", "program"):
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


def _split_tables(n_tables: int, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    """(train, validation, test) table counts that `split_dataset` cuts from `n_tables` distinct tables."""
    n_test = max(1, round(n_tables * test_fraction))
    n_val = round(n_tables * val_fraction)
    return n_tables - n_test - n_val, n_val, n_test


def min_byod_records(*, val_fraction: float = 0.15, test_fraction: float = 0.2) -> dict[str, int]:
    """The smallest BYOD set `split_dataset` always accepts when every table carries the same number of questions:
    `MIN_RECORDS` training questions on the training tables and (when `val_fraction` > 0) at least one validation
    table, after at least one test table is held out. With the default fractions that is 12 questions: one per
    table over 12 tables (split 8 / 2 / 2), or two per table over 6 tables (split 4 / 1 / 1)."""
    best: dict[str, int] | None = None
    for per_table in range(1, MIN_RECORDS + 1):
        for n in range(2, MAX_RECORDS + 1):
            train, val, test = _split_tables(n, val_fraction, test_fraction)
            if train >= 1 and train * per_table >= MIN_RECORDS and (val >= 1 or val_fraction == 0):
                if best is None or n * per_table < best["total"]:
                    best = {"total": n * per_table, "tables": n, "questions_per_table": per_table,
                            "train_tables": train, "validation_tables": val, "test_tables": test}
                break
    if best is None:
        raise ValueError("no dataset size satisfies these fractions")
    return best


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
    _n_train, n_val, n_test = _split_tables(n, val_fraction, test_fraction)
    need = min_byod_records(val_fraction=val_fraction, test_fraction=test_fraction)
    advice = (
        f"supply at least {need['total']} questions over {need['tables']} tables "
        f"({need['questions_per_table']} per table), or more tables"
    )
    if n - n_test - n_val < 1 or (val_fraction > 0 and n_val < 1):
        raise ValueError(f"{n} distinct tables are too few to split into train/validation/test — {advice}")
    out = {"train": [], "validation": [], "test": []}
    for i, key in enumerate(keys):
        name = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
        out[name].extend(by_table[key])
    if len(out["train"]) < MIN_RECORDS:
        raise ValueError(
            f"the split leaves {len(out['train'])} training questions on {n - n_test - n_val} training table(s); "
            f"at least {MIN_RECORDS} are required — {advice}"
        )
    return out


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read records from a JSONL file (one record per line) or a JSON list; validation happens downstream. A UTF-8
    byte-order mark is accepted, and a malformed line is refused with its line number in the file."""
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
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
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{source.name} line {number}: not valid JSON ({exc.msg} at column {exc.colno}); each line must be "
                "one record object"
            ) from None
        if not isinstance(row, Mapping):
            raise ValueError(f"{source.name} line {number}: each JSONL line must be a record object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{source.name} holds no records")
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
    """A human-readable summary (id, category, source, aggregation, question, FinQA programme, denotation), not the BYOD format."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "category", "source", "aggregation", "question", "program", "denotation"])
        for r in records:
            d = r["answer"]["denotation"]
            writer.writerow([r["id"], r.get("category", ""), r.get("source", ""), r["answer"]["aggregation"], r["question"], r.get("program", ""), json.dumps(d) if isinstance(d, list) else d])
    return out
