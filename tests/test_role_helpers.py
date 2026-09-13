"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from tapas_table_qa_pipeline import (
    AGGREGATIONS,
    CELL_THRESHOLD,
    DECISION_RULE,
    INPUT_SCHEMA,
    MAX_CELL_CHARS,
    MAX_COLUMNS,
    MAX_QUERY_CHARS,
    MAX_ROWS,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)

TABLE = {
    "City": ["Manila", "Cebu", "Davao", "Baguio"],
    "Population (2020)": ["1,846,513", "964,169", "1,776,949", "366,358"],
    "Region": ["NCR", "Region VII", "Region XI", "CAR"],
}
QUERIES = [
    "Which city is in Region VII?",
    "How many cities are listed?",
    "Total population of Manila and Davao?",
]


def _result(cells: list[str], aggregation: str, numeric: float | None) -> dict:
    return {
        "cells": cells,
        "coordinates": [[0, 0]] * len(cells),
        "aggregation": aggregation,
        "answer": ", ".join(cells),
        "numeric_answer": numeric,
        "unparsed_cells": [],
        "n_tokens": 60,
        "rows_kept": 4,
        "truncated": False,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(TABLE, QUERIES, names=["lookup", "count", "sum"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["rows"] == [1, MAX_ROWS]
    assert manifest["schema"]["columns"] == [1, MAX_COLUMNS]
    assert manifest["schema"]["tokens"] == [1, MAX_TOKENS]
    assert manifest["schema"]["query_chars"] == [1, MAX_QUERY_CHARS]
    assert manifest["schema"]["cell_chars"] == [0, MAX_CELL_CHARS]
    assert manifest["schema"]["aggregations"] == list(AGGREGATIONS)
    assert manifest["schema"]["cell_threshold"] == CELL_THRESHOLD
    assert manifest["schema"]["decision_rule"] == DECISION_RULE
    assert manifest["table"] == {"columns": list(TABLE), "n_rows": 4, "n_columns": 3, "numeric_cells": 4}
    assert manifest["inputs"] == [
        {"id": "lookup", "chars": len(QUERIES[0]), "query": QUERIES[0]},
        {"id": "count", "chars": len(QUERIES[1]), "query": QUERIES[1]},
        {"id": "sum", "chars": len(QUERIES[2]), "query": QUERIES[2]},
    ]
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids_and_row_form() -> None:
    rows = [dict(zip(TABLE, values, strict=True)) for values in zip(*TABLE.values(), strict=True)]
    manifest = validate_inputs(rows, ["One question?"])
    assert [entry["id"] for entry in manifest["inputs"]] == ["query-0"]
    assert manifest["table"]["n_rows"] == 4 and manifest["table"]["columns"] == list(TABLE)


def test_validate_inputs_rejects_like_the_core_method() -> None:
    with pytest.raises(TypeError, match="table must be"):
        validate_inputs("not a table", ["q"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"cell\[0\]\[0\] must be str"):
        validate_inputs({"n": [1]}, ["q"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match=f"MAX_ROWS={MAX_ROWS}"):
        validate_inputs({"n": ["x"] * (MAX_ROWS + 1)}, ["q"])
    with pytest.raises(ValueError, match=f"MAX_COLUMNS={MAX_COLUMNS}"):
        validate_inputs({f"c{i}": ["x"] for i in range(MAX_COLUMNS + 1)}, ["q"])
    with pytest.raises(ValueError, match="unequal lengths"):
        validate_inputs({"a": ["1", "2"], "b": ["1"]}, ["q"])
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs(TABLE, "a bare string")
    with pytest.raises(ValueError, match="at least one item"):
        validate_inputs(TABLE, [])
    with pytest.raises(ValueError, match="query is empty"):
        validate_inputs(TABLE, ["  "])
    with pytest.raises(ValueError, match=f"MAX_QUERY_CHARS={MAX_QUERY_CHARS}"):
        validate_inputs(TABLE, ["q" * (MAX_QUERY_CHARS + 1)])
    with pytest.raises(ValueError, match="names must have one entry per query"):
        validate_inputs(TABLE, ["q"], names=["a", "b"])


def test_evaluation_report_is_not_measurable_without_golds() -> None:
    results = [_result(["Cebu"], "NONE", None), _result(list(TABLE["City"]), "COUNT", 4.0)]
    report = evaluation_report(results)
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_results"] == 2
    assert report["aggregations"] == ["NONE", "COUNT"]
    assert report["sample_kind"] == "synthetic"
    assert report["decision_rule"] == DECISION_RULE
    assert "no gold denotation" in report["reason"]
    assert "denotation_accuracy" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_golds() -> None:
    results = [
        _result(["Cebu"], "NONE", None),
        _result(list(TABLE["City"]), "COUNT", 4.0),
        _result(["1,846,513", "1,776,949"], "SUM", 3_623_462.0),
    ]
    report = evaluation_report(results, ["Cebu", 4, "3,623,462"], sample_kind="BYOD upload")
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "BYOD upload"
    assert report["metrics"] == [
        {
            "id": "denotation_accuracy",
            "value": 1.0,
            "n": 3,
            "estimation": "single sample, no dispersion estimate",
        }
    ]
    assert "not a benchmark" in report["reason"]
    partial = evaluation_report(results, ["Davao", 4, 0])
    assert partial["metrics"][0]["value"] == pytest.approx(1 / 3)
    single = evaluation_report(results[0], ["cebu"])
    assert single["n_results"] == 1 and single["metrics"][0]["value"] == 1.0
    with pytest.raises(ValueError):
        evaluation_report(results, ["only one"])
