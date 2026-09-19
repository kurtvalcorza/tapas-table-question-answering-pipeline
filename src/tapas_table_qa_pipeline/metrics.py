"""Corpus-level table-QA measures and two non-neural baselines, in plain Python.

``denotation_metrics`` scores pipeline results against records of the dataset contract (``samples.py``):
denotation accuracy (WTQ-style match of the pipeline's answer against the record's denotation), aggregation
accuracy (the predicted operator equals the record's), cell accuracy (the selected coordinates equal the
record's as a set), each overall, per ``category`` and per gold aggregation. The baselines answer through the
same result shape so they are scored by the same function.
"""

# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import AGGREGATIONS, DECISION_RULE, compute_numeric_answer, denotation_match

METRIC_DEFINITIONS = {
    "accuracy": (
        "fraction of questions whose pipeline answer matches the record's denotation (numeric to 1e-6 for "
        "an operator, otherwise the same multiset of cell strings after lower-casing); in 0..1"
    ),
    "aggregation_accuracy": "fraction of questions whose predicted operator equals the record's; in 0..1",
    "cell_accuracy": "fraction of questions whose selected coordinates equal the record's as a set; in 0..1",
}
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(str(text).lower()))


def _result_matches(result: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    answer = record["answer"]
    gold = answer["denotation"]
    denotation = denotation_match(result, gold if not isinstance(gold, list) else list(gold))
    aggregation = result.get("aggregation") == answer["aggregation"]
    predicted = sorted(tuple(int(v) for v in c) for c in result.get("coordinates", []))
    cells = predicted == sorted(tuple(int(v) for v in c) for c in answer["coordinates"])
    return bool(denotation), bool(aggregation), bool(cells)


def denotation_metrics(
    results: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score one result per record; see METRIC_DEFINITIONS. Raises when the lengths differ or nothing is scored."""
    if len(results) != len(records) or not results:
        raise ValueError("results and records must be non-empty and the same length")
    rows = []
    for result, record in zip(results, records, strict=True):
        d, a, c = _result_matches(result, record)
        rows.append(
            {
                "id": record["id"],
                "category": record.get("category"),
                "gold_aggregation": record["answer"]["aggregation"],
                "aggregation": result.get("aggregation"),
                "match": d,
                "aggregation_match": a,
                "cell_match": c,
            }
        )

    def _summary(items: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        n = len(items)
        return {
            "n": n,
            "accuracy": sum(r["match"] for r in items) / n,
            "aggregation_accuracy": sum(r["aggregation_match"] for r in items) / n,
            "cell_accuracy": sum(r["cell_match"] for r in items) / n,
        }

    categories = sorted({r["category"] for r in rows if r["category"] is not None})
    return {
        **_summary(rows),
        "per_category": {c: _summary([r for r in rows if r["category"] == c]) for c in categories},
        "per_aggregation": {
            a: _summary([r for r in rows if r["gold_aggregation"] == a])
            for a in AGGREGATIONS
            if any(r["gold_aggregation"] == a for r in rows)
        },
        "predictions": rows,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def _baseline_result(columns: Sequence[str], rows: Sequence[Sequence[str]], coords, aggregation: str) -> dict[str, Any]:
    cells = [rows[r][c] for r, c in coords]
    value, unparsed = compute_numeric_answer(cells, aggregation)
    return {
        "cells": cells,
        "coordinates": [list(c) for c in coords],
        "aggregation": aggregation,
        "answer": (f"{aggregation} > " if aggregation != "NONE" else "") + ", ".join(cells),
        "numeric_answer": value,
        "unparsed_cells": unparsed,
        "decision_rule": DECISION_RULE,
    }


def first_cell_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Floor: answer every question with the first cell of the first column and NONE."""
    results = []
    for record in records:
        columns = list(record["table"])
        rows = [[record["table"][h][i] for h in columns] for i in range(len(record["table"][columns[0]]))]
        results.append(_baseline_result(columns, rows, [(0, 0)], "NONE"))
    return {**denotation_metrics(results, records), "baseline": "first cell of the first column, NONE"}


def keyword_lookup_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A non-neural heuristic: the answer column is the header sharing the most words with the question, the
    answer rows are those whose other cells appear verbatim in the question (else the first row), and the
    operator is read from the question's wording (``how many`` → COUNT, ``total``/``sum`` → SUM,
    ``average`` → AVERAGE, else NONE)."""
    results = []
    for record in records:
        columns = list(record["table"])
        rows = [[record["table"][h][i] for h in columns] for i in range(len(record["table"][columns[0]]))]
        question = record["question"].lower()
        q_words = _words(question)
        overlap = [len(_words(h) & q_words) for h in columns]
        sel = max(range(len(columns)), key=lambda i: (overlap[i], -i))
        hits = [
            r
            for r, row in enumerate(rows)
            if any(c != sel and len(str(cell).strip()) >= 2 and str(cell).strip().lower() in question for c, cell in enumerate(row))
        ]
        if not hits:
            hits = [0]
        if "how many" in question or question.startswith("count"):
            aggregation = "COUNT"
        elif "total" in question or "sum " in question:
            aggregation = "SUM"
        elif "average" in question or "mean " in question:
            aggregation = "AVERAGE"
        else:
            aggregation = "NONE"
        results.append(_baseline_result(columns, rows, [(r, sel) for r in hits], aggregation))
    return {**denotation_metrics(results, records), "baseline": "header-overlap column, verbatim-cell rows, wording-based operator"}
