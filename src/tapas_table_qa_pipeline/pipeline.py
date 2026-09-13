"""Table question answering over the pinned ``google/tapas-large-finetuned-wtq`` checkpoint.

Weights load only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed,
from the Hugging Face Hub at the pinned revision. One task method: ``answer(table, query)`` — the model
selects table cells and one aggregation operator (NONE/SUM/AVERAGE/COUNT) exactly as the upstream
``TapasTokenizer.convert_logits_to_predictions`` does; the numeric value for SUM/AVERAGE/COUNT is then
computed *by this module* from the selected cell strings and labelled as such.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "google/tapas-large-finetuned-wtq"
MODEL_REVISION = "f58317ab2577d17647d9acafa790c744a0388b30"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "tapas-large-wtq"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Ceilings. MAX_ROWS/MAX_COLUMNS are ``max_num_rows``/``max_num_columns`` in the pinned config.json;
# MAX_TOKENS is ``model_max_length`` in tokenizer_config.json and the fine-tuning sequence length (README).
# A table over MAX_ROWS or MAX_COLUMNS is rejected. A table that fits those but tokenises past MAX_TOKENS
# is TRUNCATED by the tokenizer (``drop_rows_to_fit``): cell texts are first capped at a common token
# count, then trailing rows are dropped until the flattened table fits; the result reports ``rows_kept``.
MAX_ROWS = 64
MAX_COLUMNS = 32
MAX_TOKENS = 512
MAX_QUERY_CHARS = 500
MAX_CELL_CHARS = 200
# ``aggregation_labels`` in config.json, index order; the aggregation head is an argmax over these four.
AGGREGATIONS = ("NONE", "SUM", "AVERAGE", "COUNT")
# ``cell_classification_threshold`` default in the upstream convert_logits_to_predictions: a cell is
# selected when the mean sigmoid probability over its tokens exceeds this value.
CELL_THRESHOLD = 0.5
DECISION_RULE = (
    f"cell selected when its mean token sigmoid probability > CELL_THRESHOLD={CELL_THRESHOLD}; aggregation "
    "operator = argmax over the four aggregation logits; numeric answer computed by the pipeline from the "
    "selected cells (COUNT = number of cells; SUM/AVERAGE parse each cell as a number, else None)"
)
NUMERIC_ANSWER_SOURCE = "computed by the pipeline from the selected cells, not emitted by the model"
_NUMBER = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _check_cell(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{where} must be str (stringify numbers yourself), got {type(value).__name__}")
    if len(value) > MAX_CELL_CHARS:
        raise ValueError(f"{where} has {len(value)} chars; ceiling is MAX_CELL_CHARS={MAX_CELL_CHARS}")
    return value


def _check_table(table: Any) -> tuple[list[str], list[list[str]]]:
    """Normalise ``{column: [cells]}`` or ``[{column: cell}, ...]`` to (columns, rows); raise on the
    first violated ceiling. Every header and cell must already be a str: TAPAS tokenises text only."""
    if isinstance(table, Mapping):
        columns = [_check_cell(name, f"column name {name!r}") for name in table]
        for name, col in table.items():
            if isinstance(col, str | bytes) or not isinstance(col, Sequence):
                raise TypeError(f"column {name!r} must be a list of str cells, got {type(col).__name__}")
        lengths = {len(col) for col in table.values()}
        if len(columns) and len(lengths) != 1:
            raise ValueError(
                f"columns have unequal lengths {sorted(lengths)}; every column needs one cell per row"
            )
        n_rows = lengths.pop() if lengths else 0
        rows = [[table[name][i] for name in columns] for i in range(n_rows)]
    elif isinstance(table, Sequence) and not isinstance(table, str | bytes):
        if not table or not all(isinstance(row, Mapping) for row in table):
            raise TypeError(
                "table must be a non-empty {column: [cells]} mapping or a list of {column: cell} rows"
            )
        columns = [_check_cell(name, f"column name {name!r}") for name in table[0]]
        rows = []
        for i, row in enumerate(table):
            if list(row) != columns:
                raise ValueError(f"row {i} has columns {list(row)}; every row must carry exactly {columns}")
            rows.append([row[name] for name in columns])
    else:
        raise TypeError("table must be a {column: [cells]} mapping or a list of {column: cell} rows")
    if not 1 <= len(columns) <= MAX_COLUMNS:
        raise ValueError(f"table has {len(columns)} columns; ceiling is 1..MAX_COLUMNS={MAX_COLUMNS}")
    if not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError(f"table has {len(rows)} rows; ceiling is 1..MAX_ROWS={MAX_ROWS}")
    checked = [[_check_cell(v, f"cell[{r}][{c}]") for c, v in enumerate(row)] for r, row in enumerate(rows)]
    return columns, checked


def _check_query(query: Any) -> str:
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")
    if not query.strip():
        raise ValueError("query is empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query has {len(query)} chars; ceiling is MAX_QUERY_CHARS={MAX_QUERY_CHARS}")
    return query


def parse_number(cell: str) -> float | None:
    """Parse one cell string as a number (optional sign, thousands commas, decimals, trailing %) or None."""
    text = cell.strip()
    if not _NUMBER.match(text):
        return None
    return float(text.rstrip("%").replace(",", ""))


def compute_numeric_answer(cells: Sequence[str], aggregation: str) -> tuple[float | None, list[str]]:
    """The pipeline-computed value for an aggregation: (value, cells that did not parse as numbers)."""
    if aggregation == "COUNT":
        return float(len(cells)), []
    if aggregation not in ("SUM", "AVERAGE") or not cells:
        return None, []
    parsed = [(cell, parse_number(cell)) for cell in cells]
    unparsed = [cell for cell, value in parsed if value is None]
    if unparsed:
        return None, unparsed
    values = [value for _, value in parsed if value is not None]
    return (sum(values) if aggregation == "SUM" else sum(values) / len(values)), []


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def denotation_match(result: Mapping[str, Any], gold: str | float | int | Sequence[str]) -> bool:
    """WTQ-style denotation match: a numeric gold is compared with the pipeline's numeric answer (or the
    single selected cell parsed as a number) to 1e-6; otherwise the selected cells and the gold strings
    must be the same multiset after lower-casing and whitespace collapse."""
    cells = [str(c) for c in result.get("cells", [])]
    if isinstance(gold, bool):
        raise TypeError("gold must be a number, a string or a sequence of strings")
    if isinstance(gold, int | float):
        gold_number: float | None = float(gold)
    else:
        gold_number = parse_number(gold) if isinstance(gold, str) else None
    predicted = result.get("numeric_answer")
    if predicted is None and len(cells) == 1 and result.get("aggregation") == "NONE":
        predicted = parse_number(cells[0])
    if gold_number is not None and predicted is not None:
        return abs(float(predicted) - gold_number) <= 1e-6
    gold_cells = [gold] if isinstance(gold, str) else [] if isinstance(gold, int | float) else list(gold)
    return sorted(_normalise(c) for c in cells) == sorted(_normalise(str(g)) for g in gold_cells)


def denotation_accuracy(results: Sequence[Mapping[str, Any]], golds: Sequence[Any]) -> float:
    """Fraction of (result, gold) pairs whose denotations match; raises when the lengths differ."""
    if len(results) != len(golds) or not results:
        raise ValueError("results and golds must be non-empty and the same length")
    return sum(denotation_match(r, g) for r, g in zip(results, golds, strict=True)) / len(results)


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "one table as {column: [cells]} or [{column: cell}, ...] with every header and cell a str, plus one "
        "non-empty question str; the model answers one question per call"
    ),
    "rows": [1, MAX_ROWS],
    "columns": [1, MAX_COLUMNS],
    "tokens": [1, MAX_TOKENS],
    "query_chars": [1, MAX_QUERY_CHARS],
    "cell_chars": [0, MAX_CELL_CHARS],
    "aggregations": list(AGGREGATIONS),
    "cell_threshold": CELL_THRESHOLD,
    "decision_rule": DECISION_RULE,
    "preprocessing": (
        "the table is flattened to [CLS] question [SEP] header row + data rows [SEP], lower-cased WordPiece; "
        "a flattened sequence over MAX_TOKENS is truncated by drop_rows_to_fit (cell texts capped at a "
        "common token count, then trailing rows dropped) and the result reports truncated, rows_kept and "
        "tokens_before_truncation; cells stay "
        "strings for the tokenizer and are parsed back to numbers only for the pipeline-computed "
        "SUM/AVERAGE answer"
    ),
}


def validate_inputs(
    table: Mapping[str, Sequence[str]] | Sequence[Mapping[str, str]],
    queries: Sequence[str],
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, table and per-query observations, verdict).

    Rejection raises exactly as ``answer`` would: both route through ``_check_table`` and ``_check_query``.
    ``answer`` takes one query per call, so ``queries`` is the batch the notebook loops over against the
    same table. The token ceiling (``MAX_TOKENS``) needs the loaded tokenizer, so it is not observable here;
    ``answer`` reports ``rows_kept`` and ``truncated`` after tokenisation.
    """
    columns, rows = _check_table(table)
    if isinstance(queries, str | bytes) or not isinstance(queries, Sequence):
        raise TypeError("queries must be a sequence of str, not a single string")
    if not queries:
        raise ValueError("queries must hold at least one item")
    checked = [_check_query(q) for q in queries]
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per query")
    return {
        "schema": dict(INPUT_SCHEMA),
        "table": {
            "columns": columns,
            "n_rows": len(rows),
            "n_columns": len(columns),
            "numeric_cells": sum(parse_number(cell) is not None for row in rows for cell in row),
        },
        "inputs": [
            {"id": names[i] if names else f"query-{i}", "chars": len(q), "query": q}
            for i, q in enumerate(checked)
        ],
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    results: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    golds: Sequence[Any] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``golds`` (one gold denotation per result: a number, a string or a list of cell strings) the report
    carries ``denotation_accuracy`` as sample-sanity evidence; without them the verdict is ``not-measurable``
    and the report says what labelled data would make the task measurable.
    """
    items = [results] if isinstance(results, Mapping) else list(results)
    base = {
        "task": "table question answering (cell selection + aggregation, WikiTableQuestions fine-tune)",
        "decision_rule": DECISION_RULE,
        "sample_kind": sample_kind,
        "n_results": len(items),
        "aggregations": [r.get("aggregation") for r in items],
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if golds is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no gold denotation was supplied for the evaluated questions",
            "needs": (
                "one gold denotation per question (the cell strings, or the number an aggregation should "
                "produce) over enough questions from the deployment's own tables to state a dispersion, "
                "scored with denotation_accuracy; the upstream WTQ dev figure is not reproduced here"
            ),
        }
    return {
        **base,
        "metrics": [
            {
                "id": "denotation_accuracy",
                "value": denotation_accuracy(items, list(golds)),
                "n": len(items),
                "estimation": "single sample, no dispersion estimate",
            }
        ],
        "verdict": "sample-sanity",
        "reason": (
            f"{len(items)} question(s) with author-supplied gold denotations on one sample table; not a "
            "benchmark"
        ),
        "needs": (
            "a labelled table-question set from the deployment domain for any generalisable accuracy claim"
        ),
    }


@dataclass
class TAPASTableQAPipeline:
    """``_runner(columns, rows, query)`` -> ``{coordinates: [(row, col)], aggregation_index,
    aggregation_logits, n_tokens, full_tokens, rows_kept}`` where coordinates index data rows (0 = first row
    below the header) and full_tokens is the untrimmed length; injectable so tests run offline."""

    _runner: Callable[[list[str], list[list[str]], str], Mapping[str, Any]]
    device: str = "cpu"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> TAPASTableQAPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, kwargs, source = str(root), dict(local_files_only=True), "local-snapshot"
        elif allow_download:
            location, kwargs, source = MODEL_ID, dict(revision=MODEL_REVISION), "hf-hub"
        else:
            raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import TapasForQuestionAnswering, TapasTokenizer

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        import pandas as pd  # TapasTokenizer takes a DataFrame; imported after the refusal like the others
        tokenizer = TapasTokenizer.from_pretrained(location, trust_remote_code=False, **kwargs)
        model = TapasForQuestionAnswering.from_pretrained(
            location, dtype=torch.float32, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()

        class _PositionalRows(pd.DataFrame):
            """transformers 4.57.6's TapasTokenizer reads each ``iterrows()`` row by integer position
            (``row[col_index]``), which pandas 3 no longer accepts on a str-labelled Series; rows are yielded
            re-indexed by position so the tokenizer's numeric-value annotation runs unchanged."""

            @property
            def _constructor(self):
                return _PositionalRows

            def iterrows(self):
                for index, row in super().iterrows():
                    yield index, row.reset_index(drop=True)

        def runner(columns: list[str], rows: list[list[str]], query: str) -> dict[str, Any]:
            # dtype=object: pandas 3's default pyarrow str dtype cannot hold the tokenizer's Cell objects
            frame = _PositionalRows(rows, columns=columns, dtype=object)
            encoded = tokenizer(
                table=frame, queries=[query], truncation="drop_rows_to_fit", max_length=MAX_TOKENS,
                padding="max_length", return_tensors="pt",
            )
            with torch.inference_mode():
                out = model(**{k: v.to(resolved_device) for k, v in encoded.items()})
            coordinates, agg = tokenizer.convert_logits_to_predictions(
                encoded,
                out.logits.cpu(),
                out.logits_aggregation.cpu(),
                cell_classification_threshold=CELL_THRESHOLD,
            )
            # Untrimmed length ([CLS] query [SEP] header + cells): when it exceeds n_tokens the tokenizer
            # trimmed cell text and/or dropped rows to fit MAX_TOKENS.
            flat = [query, *columns, *(cell for row in rows for cell in row)]
            return {
                "coordinates": [(int(r), int(c)) for r, c in coordinates[0]] if coordinates else [],
                "aggregation_index": int(agg[0]),
                "aggregation_logits": out.logits_aggregation[0].float().cpu().tolist(),
                "n_tokens": int(encoded["attention_mask"].sum()),
                "full_tokens": 2 + sum(len(tokenizer.tokenize(text)) for text in flat),
                "rows_kept": int(encoded["token_type_ids"][0, :, 2].max()),
            }

        return cls(runner, resolved_device, source)

    def answer(
        self, table: Mapping[str, Sequence[str]] | Sequence[Mapping[str, str]], query: str
    ) -> dict[str, Any]:
        """Select cells and an aggregation operator for one question; compute the numeric answer from them."""
        columns, rows = _check_table(table)
        query = _check_query(query)
        raw = self._runner(columns, rows, query)
        coordinates = [(int(r), int(c)) for r, c in raw["coordinates"]]
        if any(not (0 <= r < len(rows) and 0 <= c < len(columns)) for r, c in coordinates):
            raise RuntimeError(f"backend returned coordinates outside the {len(rows)}x{len(columns)} table")
        aggregation = AGGREGATIONS[int(raw["aggregation_index"])]
        cells = [rows[r][c] for r, c in coordinates]
        value, unparsed = compute_numeric_answer(cells, aggregation)
        rows_kept = int(raw.get("rows_kept", len(rows)))
        n_tokens = int(raw.get("n_tokens", 0))
        full_tokens = int(raw.get("full_tokens", n_tokens))
        return {
            "cells": cells,
            "coordinates": [list(c) for c in coordinates],
            "aggregation": aggregation,
            "answer": (f"{aggregation} > " if aggregation != "NONE" else "") + ", ".join(cells),
            "numeric_answer": value,
            "numeric_answer_source": NUMERIC_ANSWER_SOURCE,
            "unparsed_cells": unparsed,
            "aggregation_logits": dict(zip(AGGREGATIONS, map(float, raw["aggregation_logits"]), strict=True)),
            "n_tokens": n_tokens,
            "tokens_before_truncation": full_tokens,
            "rows_kept": rows_kept,
            "truncated": rows_kept < len(rows) or n_tokens < full_tokens,
            "decision_rule": DECISION_RULE,
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
