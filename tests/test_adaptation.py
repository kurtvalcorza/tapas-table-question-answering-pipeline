"""Offline checks of the adaptation contract: the record contract and its refusals, the TAT-QA / FinQA conversion and
draw, table-disjoint splitting, the BYOD loader, the metrics and baselines, `evaluate` through the injected
runner, the artifact-manifest checks, and the model-free refusals of `adapt` / `save_artifact`."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json

import pytest

from tapas_table_qa_pipeline import (
    AGGREGATIONS,
    ARTIFACT_FORMAT,
    CORPORA,
    DEFAULT_CACHE_DIR,
    ENCODER_LAYERS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_DIGEST,
    SAMPLE_SPLIT,
    TAPASTableQAPipeline,
    build_sample_dataset,
    check_group_disjoint,
    check_split_disjoint,
    clean_cell,
    dataset_digest,
    denotation_metrics,
    fetch_corpus,
    finqa_candidates,
    first_cell_baseline,
    keyword_lookup_baseline,
    load_byod_dataset,
    sample_candidates,
    samples,
    split_dataset,
    table_digest,
    tatqa_candidates,
    validate_dataset,
    value_correct,
    write_dataset_csv,
    write_dataset_jsonl,
)
from tapas_table_qa_pipeline import pipeline as pl

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]

TABLE = {
    "City": ["Manila", "Cebu", "Davao", "Baguio"],
    "Population": ["1,846,513", "964,169", "1,776,949", "366,358"],
    "Region": ["NCR", "Region VII", "Region XI", "CAR"],
}


def _record(i, aggregation="NONE", coords=((0, 0),), category="lookup", table=None):
    table = table or {k: list(v) for k, v in TABLE.items()}
    cols = list(table)
    cells = [table[cols[c]][r] for r, c in coords]
    if aggregation == "NONE":
        denotation = cells
    else:
        denotation, _ = pl.compute_numeric_answer(cells, aggregation)
    return {
        "id": f"r{i:03d}",
        "table": table,
        "question": f"question {i}?",
        "answer": {"aggregation": aggregation, "coordinates": [list(c) for c in coords], "denotation": denotation},
        "category": category,
    }


def _records(n=12):
    out = []
    for i in range(n):
        table = {k: list(v) for k, v in TABLE.items()}
        table["City"][0] = f"Manila {i}"  # distinct table per record
        kind = i % 4
        if kind == 0:
            out.append(_record(i, "NONE", ((1, 2),), "lookup", table))
        elif kind == 1:
            out.append(_record(i, "COUNT", ((0, 0), (1, 0)), "count", table))
        elif kind == 2:
            out.append(_record(i, "SUM", ((0, 1), (2, 1)), "sum", table))
        else:
            out.append(_record(i, "AVERAGE", ((1, 1), (3, 1)), "average", table))
    return out


class _ScriptedRunner:
    """Answers every question with fixed coordinates and aggregation index (records the calls)."""

    def __init__(self, coordinates, aggregation_index):
        self.coordinates, self.aggregation_index, self.calls = coordinates, aggregation_index, 0

    def __call__(self, columns, rows, query):
        self.calls += 1
        return {
            "coordinates": self.coordinates,
            "aggregation_index": self.aggregation_index,
            "aggregation_logits": [0.1, 0.2, 0.3, 0.4],
            "n_tokens": 60,
            "full_tokens": 60,
            "rows_kept": len(rows),
        }


# --- record contract ---------------------------------------------------------------------------------------


def test_validate_dataset_accepts_records_and_reports_counts_and_digest():
    manifest = validate_dataset(_records())
    assert manifest["n_records"] == 12 and manifest["n_tables"] == 12
    assert manifest["aggregation_counts"] == {"AVERAGE": 3, "COUNT": 3, "NONE": 3, "SUM": 3}
    assert manifest["category_counts"] == {"average": 3, "count": 3, "lookup": 3, "sum": 3}
    assert manifest["table_shape"] == {"rows": [4, 4], "columns": [3, 3]}
    assert len(manifest["digest"]) == 64 and manifest["model_id"] == MODEL_ID
    assert manifest["records"][0]["answer"]["denotation"] == ["Region VII"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.__setitem__("id", "bad id"), "id must match"),
        (lambda r: r["answer"].__setitem__("aggregation", "MAX"), "aggregation must be one of"),
        (lambda r: r["answer"].__setitem__("coordinates", []), "non-empty list"),
        (lambda r: r["answer"].__setitem__("coordinates", [[9, 0]]), "outside the"),
        (lambda r: r["answer"].__setitem__("coordinates", [[0, 0], [0, 0]]), "duplicate coordinate"),
        (lambda r: r["answer"].__setitem__("denotation", ["Cebu"]), "does not match the selected cells"),
        (lambda r: r.__setitem__("question", ""), "query is empty"),
        (lambda r: r.__setitem__("category", "x" * 40), "category must be"),
        (lambda r: r["table"].__setitem__("City", ["a", "b"]), "unequal lengths"),
    ],
)
def test_validate_dataset_refuses_malformed_records(mutate, message):
    records = _records()
    mutate(records[0])
    with pytest.raises((ValueError, TypeError), match=message):
        validate_dataset(records)


def test_validate_dataset_checks_operator_denotations_against_the_cells():
    records = _records()
    records[1]["answer"]["denotation"] = 3.0  # COUNT of two cells
    with pytest.raises(ValueError, match="denotation says"):
        validate_dataset(records)
    records = _records()
    records[2]["answer"]["coordinates"] = [[0, 0], [2, 0]]  # SUM over text cells
    with pytest.raises(ValueError, match="do not parse as numbers"):
        validate_dataset(records)
    records = _records()
    records[0]["answer"]["denotation"] = 5  # NONE with a number
    with pytest.raises(ValueError, match="list of cell strings"):
        validate_dataset(records)


def test_validate_dataset_enforces_bounds_and_unique_ids():
    with pytest.raises(ValueError, match="8..20000"):
        validate_dataset(_records(4))
    records = _records()
    records[1]["id"] = records[0]["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(records)
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})


def test_validate_dataset_refuses_before_importing_model_libraries(forbid_model_imports):
    with pytest.raises(ValueError):
        validate_dataset(_records(3))
    validate_dataset(_records())


def test_table_digest_and_dataset_digest_are_content_and_order_independent():
    a = table_digest(TABLE)
    assert a == table_digest({k: [c.upper() for c in v] for k, v in TABLE.items()})
    assert a != table_digest({**TABLE, "City": ["x", "y", "z", "w"]})
    records = _records()
    assert dataset_digest(records) == dataset_digest(list(reversed(records)))


def test_split_dataset_is_table_disjoint_and_check_split_disjoint_catches_leaks():
    records = _records(24)
    splits = split_dataset(records, seed=1)
    assert sum(len(v) for v in splits.values()) == 24 and all(splits.values())
    assert check_split_disjoint(splits) == {k: len(v) for k, v in splits.items()}
    leaked = {**splits, "test": [*splits["test"], splits["train"][0]]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaked)
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.9)


def test_split_dataset_keeps_every_question_of_a_table_together():
    shared = {k: list(v) for k, v in TABLE.items()}
    records = [{**_record(i, "NONE", ((i % 4, 2),), "lookup", shared), "id": f"shared-{i}"} for i in range(8)] + _records(8)
    splits = split_dataset(records, seed=3)
    homes = {name for name, part in splits.items() for r in part if r["id"].startswith("shared-")}
    assert len(homes) == 1


def test_byod_jsonl_and_json_round_trip(tmp_path):
    records = _records()
    path = write_dataset_jsonl(records, tmp_path / "byod.jsonl")
    loaded = load_byod_dataset(path)
    assert [r["id"] for r in loaded] == [r["id"] for r in records]
    assert validate_dataset(loaded)["digest"] == validate_dataset(records)["digest"]
    (tmp_path / "list.json").write_text(json.dumps(records), encoding="utf-8")
    assert len(load_byod_dataset(tmp_path / "list.json")) == 12
    (tmp_path / "bad.jsonl").write_text("[1, 2]\n[3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record object"):
        load_byod_dataset(tmp_path / "bad.jsonl")
    csv_path = write_dataset_csv(records, tmp_path / "train.csv")
    assert csv_path.read_text(encoding="utf-8").startswith("id,category,source,aggregation,question,program,denotation")


# --- TAT-QA / FinQA conversion and draw ----------------------------------------------------------------------


def _tatqa_context(t, answers=("20",), answer_type="span"):
    return {
        "table": {"uid": f"tab-{t}", "table": [["", "2019", "2018"], [f"Revenue {t}", "$ 10", "20"], ["Cost", "( 5 )", "7"]]},
        "questions": [{"uid": f"q-{t}", "question": f"What was revenue {t} in 2018?", "answer": list(answers), "answer_type": answer_type, "answer_from": "table"}],
    }


def _finqa_example(t, program, exe_ans, company="ACME", gold=("table_1", "table_2", "table_3"), answer=None):
    # Vertical table: one year column; rows are line items (single-column golds are possible).
    return {
        "id": f"{company}/{2000 + t}/page_1.pdf-{t}",
        "table": [["", "2009", "2008"], ["net revenue", "$ 1,000", "900"], ["costs", "( 100 )", "80"], ["other", "2,000", "70"], ["total", "2,900", "1,050"]],
        "qa": {"question": f"q{t}", "program": program, "exe_ans": exe_ans, "answer": answer if answer is not None else str(exe_ans), "gold_inds": {k: "x" for k in gold}},
    }


def test_clean_cell_rewrites_currency_and_negative_formats_only():
    assert clean_cell("$ 1,234") == "1,234" and clean_cell("( 12 )") == "-12" and clean_cell("$ -23158 ( 23158 )") == "-23158"
    assert clean_cell("12.5 %") == "12.5%" and clean_cell("( 4.5 % )") == "-4.5%" and clean_cell("net  revenue") == "net revenue"
    assert pl.parse_number(clean_cell("$ 1,234")) == 1234.0 and clean_cell("19 % ( % )") == "19 % ( % )"


def test_tatqa_candidates_keep_unique_table_spans_and_count_the_funnel():
    stats = {}
    out = tatqa_candidates([_tatqa_context(1), _tatqa_context(2, answers=("Cost",)), _tatqa_context(3, answers=("absent",)), _tatqa_context(4, answer_type="arithmetic")], stats)
    assert [r["id"] for r in out] == ["tatqa-q-1", "tatqa-q-2"]
    assert out[0]["answer"] == {"aggregation": "NONE", "coordinates": [[0, 2]], "denotation": ["20"]} and out[0]["table"]["2019"] == ["10", "-5"]
    assert out[0]["table"]["item"] == ["Revenue 1", "Cost"] and out[0]["category"] == "lookup" and out[0]["source_group"] == "tatqa:tab-1"
    assert stats == {"tatqa_table_span_questions": 3, "tatqa_after_table_filter": 3}


def test_finqa_candidates_ground_terms_in_evidence_rows_one_column_and_source_rounding():
    stats = {}
    examples = [
        _finqa_example(0, "add(1000, 2000)", 3000.0, answer="3000"),  # rows 1 and 3, column 2009: kept
        _finqa_example(1, "add(1000, 900)", 1900.0),  # one row across two columns: single-column rule rejects it
        _finqa_example(2, "add(1000, 2000), add(#0, -100), divide(#1, const_3)", 966.67),  # AVERAGE: dropped
        _finqa_example(3, "table_sum(net revenue, none)", 1900.0),  # one-row programme: spans columns
        _finqa_example(4, "add(1000, 2000)", 3000.0, gold=("table_1",)),  # 2,000 only outside the evidence rows
        _finqa_example(5, "add(1000, 2000)", 3000.0, answer="3,004"),  # disagrees with the displayed answer
        _finqa_example(6, "add(1000, 2000)", 3000.0, gold=("text_1",)),  # needs text
        _finqa_example(7, "add(1000, 80)", 1080.0, answer="1.1 thousand"),  # columns differ; rejected before rounding
        _finqa_example(8, "add(80, 70)", 150.4, answer="150.4"),  # rounding: 150 vs 150.4 exceeds half a unit
        _finqa_example(9, "add(80, 70)", 150.0, answer="150"),  # kept
    ]
    out = finqa_candidates(examples, stats)
    assert [r["id"] for r in out] == ["finqa-ACME:2000:page_1.pdf-0", "finqa-ACME:2009:page_1.pdf-9"]
    assert out[0]["answer"] == {"aggregation": "SUM", "coordinates": [[0, 1], [2, 1]], "denotation": 3000.0}
    assert out[1]["answer"]["coordinates"] == [[1, 2], [2, 2]] and out[1]["source_answer"] == "150" and out[1]["conversion_discrepancy"] == 0.0
    assert out[0]["evidence_rows"] == [0, 1, 2] and out[0]["source_group"] == "finqa:ACME/2000"
    assert stats["finqa_dropped_average"] == 1 and stats["finqa_removed_by_evidence_rows"] == 1
    assert stats["finqa_rejected_multi_column"] == 3 and stats["finqa_rejected_source_disagrees"] == 2
    assert all(len({c for _, c in r["answer"]["coordinates"]}) == 1 for r in out)


def test_source_reference_tolerance_is_half_a_unit_of_the_displayed_answer():
    assert samples._source_reference("704", 704.0) == (704.0, pytest.approx(0.5))
    assert samples._source_reference("12.5%", 12.4999) == (12.5, pytest.approx(0.05))
    assert samples._source_reference("$ 1,234.56 million", 0) == (1234.56, pytest.approx(0.005))
    value, tol = samples._source_reference("", 3.0)
    assert value == 3.0 and tol < 1e-6


def test_validate_dataset_refuses_gold_cells_in_several_columns():
    table = {"name": ["a", "b"], "x": ["1", "2"], "y": ["3", "4"]}
    record = {"id": "r0", "table": table, "question": "q", "answer": {"aggregation": "SUM", "coordinates": [[0, 1], [0, 2]], "denotation": 4.0}}
    with pytest.raises(ValueError, match="span 2 columns .*select_one_column"):
        validate_dataset([record], min_records=1)


def test_check_group_disjoint_flags_a_group_in_two_splits():
    a = {"source_group": "finqa:A/2001"}
    assert check_group_disjoint({"train": [a], "test": [{"source_group": "finqa:B/2001"}, {}]}) == {"train": 1, "test": 1}
    with pytest.raises(ValueError, match="appears in both"):
        check_group_disjoint({"train": [a], "test": [a]})


def test_build_sample_dataset_keeps_groups_disjoint_and_draws_per_category():
    contexts = [_tatqa_context(t) for t in range(40)]
    finqa = [_finqa_example(t, "add(1000, 2000)", 3000.0, company=f"C{t}") for t in range(40)]
    for i, ex in enumerate(finqa):
        ex["table"][1][0] = f"net revenue {i}"
    candidates = tatqa_candidates(contexts) + finqa_candidates(finqa)
    sizes = {"train": {"lookup": 8, "sum": 6}, "validation": 2, "test": 3}
    splits = build_sample_dataset(candidates, seed=7, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == {"train": 14, "validation": 4, "test": 6}
    assert check_split_disjoint(splits) and check_group_disjoint(splits)
    assert {r["category"] for part in splits.values() for r in part} == {"lookup", "sum"}
    with pytest.raises(ValueError, match="candidates, need"):
        build_sample_dataset(candidates, sizes={"train": {"lookup": 60}, "validation": 1, "test": 1})
    assert build_sample_dataset(candidates, seed=7, sizes=sizes) == splits


def test_fetch_corpus_refuses_a_file_that_does_not_match_the_pins(tmp_path):
    with pytest.raises(ValueError, match="bytes, pinned"):
        fetch_corpus("tatqa-train", cache_dir=tmp_path, fetcher=lambda url: b"not-a-file")
    fake = b"x" * CORPORA["tatqa-train"]["bytes"]
    assert hashlib.sha256(fake).hexdigest() != CORPORA["tatqa-train"]["sha256"]
    with pytest.raises(ValueError, match="sha256"):
        fetch_corpus("tatqa-train", cache_dir=tmp_path, fetcher=lambda url: fake)
    assert not (tmp_path / "tatqa-train.json").exists()
    assert all(spec["url"].startswith("https://") and spec["revision"] in spec["url"] for spec in CORPORA.values())


def test_default_draw_matches_the_pinned_digest_when_the_files_are_cached():
    cache = REPO_ROOT / DEFAULT_CACHE_DIR
    if not all((cache / f"{key}.json").is_file() for key in CORPORA):
        pytest.skip("TAT-QA / FinQA files not cached")
    splits = build_sample_dataset(sample_candidates({key: fetch_corpus(key, cache_dir=cache) for key in CORPORA}))
    assert {k: len(v) for k, v in splits.items()} == {k: (sum(v.values()) if isinstance(v, dict) else v * 3) for k, v in SAMPLE_SPLIT.items()}
    assert check_split_disjoint(splits)
    assert dataset_digest([r for part in splits.values() for r in part]) == SAMPLE_DIGEST


# --- metrics, baselines, evaluate ----------------------------------------------------------------------------


def test_cell_accuracy_compares_coordinate_sets_and_strict_none_follows_value_equality():
    table = {"name": ["a", "b"], "n": ["5", "7"]}
    rec = {"id": "x", "table": table, "question": "q", "answer": {"aggregation": "NONE", "coordinates": [[0, 1]], "denotation": ["5"]}}
    dup = {"cells": ["5", "5"], "coordinates": [[0, 1], [0, 1]], "aggregation": "NONE", "numeric_answer": None}
    assert denotation_metrics([dup], [rec])["cell_accuracy"] == 1.0  # a set: duplicates ignored
    numeric = {"cells": ["5.0"], "coordinates": [[1, 1]], "aggregation": "NONE", "numeric_answer": None}
    m = denotation_metrics([numeric], [rec])
    assert m["value_accuracy"] == 1.0 and m["strict_accuracy"] == 1.0 and m["cell_accuracy"] == 0.0


def test_value_and_strict_rules_score_the_final_answer():
    # Kurt's TQA-M1 ruling: value-equality regardless of operator; strict-operator separate.
    assert value_correct({"cells": ["5"], "aggregation": "NONE", "numeric_answer": None}, 5.0)  # NONE lookup equal to a SUM gold
    assert value_correct({"cells": ["5"], "aggregation": "SUM", "numeric_answer": 5.0}, ["5"])  # operator over the one gold cell
    assert not value_correct({"cells": ["Smith"], "aggregation": "COUNT", "numeric_answer": 1.0}, ["Smith"])
    assert not value_correct({"cells": ["5", "7"], "aggregation": "SUM", "numeric_answer": 12.0}, ["5", "7"])
    assert value_correct({"cells": ["Smith"], "aggregation": "NONE", "numeric_answer": None}, ["smith"])
    assert not value_correct({"cells": ["5", "7"], "aggregation": "AVERAGE", "numeric_answer": None}, 6.0)
    table = {"name": ["a", "b"], "n": ["5", "7"]}
    rec = {"id": "x", "table": table, "question": "q", "answer": {"aggregation": "SUM", "coordinates": [[0, 1]], "denotation": 5.0}, "category": "sum"}
    m = denotation_metrics([{"cells": ["5"], "coordinates": [[0, 1]], "aggregation": "NONE", "numeric_answer": None}], [rec])
    assert m["value_accuracy"] == 1.0 and m["strict_accuracy"] == 0.0 and m["aggregation_accuracy"] == 0.0 and m["cell_accuracy"] == 1.0
    assert "accuracy" not in m and set(m["definitions"]) == {"value_accuracy", "strict_accuracy", "aggregation_accuracy", "cell_accuracy"}


def test_denotation_metrics_scores_denotation_operator_and_cells_per_category():
    records = _records()
    results = []
    for r in records:
        answer = r["answer"]
        cols = list(r["table"])
        cells = [r["table"][cols[c]][row] for row, c in answer["coordinates"]]
        value, _ = pl.compute_numeric_answer(cells, answer["aggregation"])
        results.append({"cells": cells, "coordinates": answer["coordinates"], "aggregation": answer["aggregation"], "numeric_answer": value})
    perfect = denotation_metrics(results, records)
    assert perfect["value_accuracy"] == 1.0 and perfect["strict_accuracy"] == 1.0 and perfect["aggregation_accuracy"] == 1.0 and perfect["cell_accuracy"] == 1.0
    assert set(perfect["per_category"]) == {"average", "count", "lookup", "sum"}
    assert set(perfect["per_aggregation"]) <= set(AGGREGATIONS)
    results[0] = {"cells": ["Cebu"], "coordinates": [[1, 0]], "aggregation": "NONE", "numeric_answer": None}
    partial = denotation_metrics(results, records)
    assert partial["value_accuracy"] == pytest.approx(11 / 12) and partial["per_category"]["lookup"]["value_accuracy"] == pytest.approx(2 / 3)
    assert partial["strict_accuracy"] == pytest.approx(11 / 12)
    assert partial["predictions"][0]["value_match"] is False and partial["predictions"][0]["cell_match"] is False
    with pytest.raises(ValueError, match="same length"):
        denotation_metrics(results[:3], records)


def test_baselines_answer_through_the_result_shape():
    records = _records()
    floor = first_cell_baseline(records)
    assert 0.0 <= floor["value_accuracy"] <= 1.0 and floor["baseline"].startswith("first cell")
    records[0]["question"] = "Which Region is Cebu in?"
    keyword = keyword_lookup_baseline(records)
    assert keyword["predictions"][0]["value_match"] is True
    records[1]["question"] = "How many cities are in the table?"
    assert keyword_lookup_baseline(records)["predictions"][1]["aggregation"] == "COUNT"


def test_evaluate_runs_every_record_through_answer_and_scores_it():
    runner = _ScriptedRunner([(1, 2)], 0)
    pipe = TAPASTableQAPipeline(runner, "cpu", "injected")
    result = pipe.evaluate(_records())
    assert runner.calls == 12 and result["n"] == 12 and result["adapted"] is False
    assert result["per_category"]["lookup"]["value_accuracy"] == 1.0 and result["verdict"] == "measured-small-sample"
    assert result["truncated"] == 0 and "definitions" in result
    with pytest.raises(ValueError):
        pipe.evaluate([{"id": "x"}] * 8)


def test_adapt_and_save_artifact_require_a_loaded_model(forbid_model_imports):
    pipe = TAPASTableQAPipeline(_ScriptedRunner([(0, 0)], 0), "cpu", "injected")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.adapt(_records())
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.save_artifact("x")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.load_artifact("x")


# --- artifact manifest checks ---------------------------------------------------------------------------------


def _manifest(tmp_path, **overrides):
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"tensor-bytes")
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": "base-digest"},
        "adapter": {"trainable_layers": 4},
        "tensors": ["tapas.encoder.layer.23.attention.self.query.weight", "aggregation_classifier.weight"],
        "files": [{"path": "adapter.safetensors", "bytes": weights.stat().st_size, "sha256": hashlib.sha256(b"tensor-bytes").hexdigest()}],
    }
    manifest.update(overrides)
    return manifest


def test_check_artifact_manifest_accepts_a_consistent_manifest_and_refuses_each_deviation(tmp_path):
    pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="format"):
        pl._check_artifact_manifest(_manifest(tmp_path, format="other"), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trained on"):
        pl._check_artifact_manifest(_manifest(tmp_path, base={"model_id": "x", "revision": MODEL_REVISION, "weight_sha256": "base-digest"}), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="base weight digest"):
        pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "another-digest")
    bad = _manifest(tmp_path)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        pl._check_artifact_manifest(bad, tmp_path, "base-digest")
    with pytest.raises(ValueError, match="encoder blocks or the heads"):
        pl._check_artifact_manifest(_manifest(tmp_path, tensors=["tapas.embeddings.word_embeddings.weight"]), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trainable_layers"):
        pl._check_artifact_manifest(_manifest(tmp_path, adapter={"trainable_layers": ENCODER_LAYERS + 1}), tmp_path, "base-digest")


def test_trainable_names_selects_the_last_blocks_and_the_heads():
    class _Param:
        def numel(self):
            return 1

    class _Model:
        class tapas:  # noqa: N801
            class encoder:  # noqa: N801
                layer = list(range(24))

        def named_parameters(self):
            names = [f"tapas.encoder.layer.{i}.weight" for i in range(24)] + ["tapas.embeddings.w", "tapas.pooler.w", "output_weights", "output_bias", "column_output_weights", "column_output_bias", "aggregation_classifier.weight", "aggregation_classifier.bias"]
            return [(n, _Param()) for n in names]

    names = pl._trainable_names(_Model(), 2)
    assert names == ["tapas.encoder.layer.22.weight", "tapas.encoder.layer.23.weight", "output_weights", "output_bias", "column_output_weights", "column_output_bias", "aggregation_classifier.weight", "aggregation_classifier.bias"]
    assert len(pl._trainable_names(_Model(), 0)) == 6
    with pytest.raises(ValueError, match="trainable_layers"):
        pl._trainable_names(_Model(), 25)
