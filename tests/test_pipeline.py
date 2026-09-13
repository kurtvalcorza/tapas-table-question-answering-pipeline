import hashlib
import json
import re
from pathlib import Path

import pytest

from tapas_table_qa_pipeline import (
    AGGREGATIONS,
    DEFAULT_WEIGHTS_DIR,
    MAX_CELL_CHARS,
    MAX_COLUMNS,
    MAX_QUERY_CHARS,
    MAX_ROWS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    NUMERIC_ANSWER_SOURCE,
    TAPASTableQAPipeline,
    compute_numeric_answer,
    denotation_accuracy,
    denotation_match,
    parse_number,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]
TABLE = {
    "City": ["Manila", "Cebu", "Davao", "Baguio"],
    "Population (2020)": ["1,846,513", "964,169", "1,776,949", "366,358"],
    "Region": ["NCR", "Region VII", "Region XI", "CAR"],
}
ROWS = [dict(zip(TABLE, values, strict=True)) for values in zip(*TABLE.values(), strict=True)]


class _FakeRunner:
    """Scripted backend: fixed coordinates + aggregation index, records what it was given."""

    def __init__(self, coordinates, aggregation_index, rows_kept=None, full_tokens=60):
        self.coordinates, self.aggregation_index, self.rows_kept = coordinates, aggregation_index, rows_kept
        self.full_tokens, self.calls = full_tokens, []

    def __call__(self, columns, rows, query):
        self.calls.append((columns, rows, query))
        return {
            "coordinates": self.coordinates,
            "aggregation_index": self.aggregation_index,
            "aggregation_logits": [0.1, 0.2, 0.3, 0.4],
            "n_tokens": 60,
            "full_tokens": self.full_tokens,
            "rows_kept": len(rows) if self.rows_kept is None else self.rows_kept,
        }


def _pipeline(coordinates, aggregation_index, rows_kept=None, full_tokens=60) -> TAPASTableQAPipeline:
    runner = _FakeRunner(coordinates, aggregation_index, rows_kept, full_tokens)
    return TAPASTableQAPipeline(runner, "cpu", "injected")


def _write_snapshot(root: Path, payload: bytes = b"weights") -> Path:
    (root / "model.safetensors").write_bytes(payload)
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "google/tapas-large-finetuned-wtq"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert AGGREGATIONS == ("NONE", "SUM", "AVERAGE", "COUNT")


def test_identity_matches_local_manifest_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    result = verify_snapshot(_write_snapshot(tmp_path).parent)
    assert result["revision"] == MODEL_REVISION and result["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    manifest["files"][0]["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_size_missing_file_and_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        TAPASTableQAPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    assert len(verify_snapshot(tmp_path)["files"]) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_answer_rejects_bad_tables_before_the_backend_runs():
    pipe = _pipeline([(0, 0)], 0)
    with pytest.raises(TypeError, match="table must be"):
        pipe.answer("City,Population", "q")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="table must be"):
        pipe.answer([], "q")
    with pytest.raises(TypeError, match="must be a list of str cells"):
        pipe.answer({"City": "Manila"}, "q")  # type: ignore[dict-item]
    with pytest.raises(TypeError, match=r"cell\[1\]\[1\] must be str"):
        pipe.answer({"City": ["a", "b"], "Pop": ["1", 2]}, "q")  # type: ignore[list-item]
    with pytest.raises(TypeError, match="column name 3 must be str"):
        pipe.answer({3: ["a"]}, "q")  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="unequal lengths"):
        pipe.answer({"City": ["a", "b"], "Pop": ["1"]}, "q")
    with pytest.raises(ValueError, match="every row must carry exactly"):
        pipe.answer([{"a": "1"}, {"b": "2"}], "q")
    with pytest.raises(ValueError, match=f"MAX_COLUMNS={MAX_COLUMNS}"):
        pipe.answer({f"c{i}": ["x"] for i in range(MAX_COLUMNS + 1)}, "q")
    with pytest.raises(ValueError, match=f"MAX_ROWS={MAX_ROWS}"):
        pipe.answer({"c": ["x"] * (MAX_ROWS + 1)}, "q")
    with pytest.raises(ValueError, match="1..MAX_ROWS"):
        pipe.answer({"c": []}, "q")
    with pytest.raises(ValueError, match=f"MAX_CELL_CHARS={MAX_CELL_CHARS}"):
        pipe.answer({"c": ["x" * (MAX_CELL_CHARS + 1)]}, "q")
    assert pipe._runner.calls == []


def test_answer_rejects_bad_queries_before_the_backend_runs():
    pipe = _pipeline([(0, 0)], 0)
    with pytest.raises(TypeError, match="query must be str"):
        pipe.answer(TABLE, 42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query is empty"):
        pipe.answer(TABLE, "   ")
    with pytest.raises(ValueError, match=f"MAX_QUERY_CHARS={MAX_QUERY_CHARS}"):
        pipe.answer(TABLE, "q" * (MAX_QUERY_CHARS + 1))
    assert pipe._runner.calls == []


def test_answer_output_fields_for_a_cell_lookup():
    pipe = _pipeline([(1, 0)], 0)
    result = pipe.answer(TABLE, "Which city is in Region VII?")
    assert result["cells"] == ["Cebu"] and result["coordinates"] == [[1, 0]]
    assert result["aggregation"] == "NONE" and result["answer"] == "Cebu"
    assert result["numeric_answer"] is None and result["unparsed_cells"] == []
    assert result["numeric_answer_source"] == NUMERIC_ANSWER_SOURCE
    assert result["aggregation_logits"] == {"NONE": 0.1, "SUM": 0.2, "AVERAGE": 0.3, "COUNT": 0.4}
    assert result["n_tokens"] == 60 and result["tokens_before_truncation"] == 60
    assert result["rows_kept"] == 4 and result["truncated"] is False
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
    assert result["device"] == "cpu" and result["source"] == "injected"
    columns, rows, query = pipe._runner.calls[0]
    assert columns == list(TABLE) and rows[1] == ["Cebu", "964,169", "Region VII"]
    assert query == "Which city is in Region VII?"


def test_answer_accepts_row_form_and_computes_sum_count_average():
    total = _pipeline([(0, 1), (2, 1)], 1).answer(ROWS, "Total population of Manila and Davao?")
    assert total["answer"] == "SUM > 1,846,513, 1,776,949"
    assert total["numeric_answer"] == pytest.approx(3_623_462.0)
    count = _pipeline([(0, 0), (1, 0), (2, 0), (3, 0)], 3).answer(ROWS, "How many cities are listed?")
    assert count["aggregation"] == "COUNT" and count["numeric_answer"] == 4.0
    assert count["cells"] == list(TABLE["City"])
    avg = _pipeline([(1, 1), (3, 1)], 2).answer(ROWS, "Average population of Cebu and Baguio?")
    assert avg["aggregation"] == "AVERAGE" and avg["numeric_answer"] == pytest.approx((964_169 + 366_358) / 2)


def test_answer_reports_unparsable_cells_and_truncation():
    result = _pipeline([(0, 2), (1, 2)], 1, rows_kept=2).answer(TABLE, "Sum of regions?")
    assert result["aggregation"] == "SUM" and result["numeric_answer"] is None
    assert result["unparsed_cells"] == ["NCR", "Region VII"]
    assert result["rows_kept"] == 2 and result["truncated"] is True
    trimmed = _pipeline([(0, 0)], 0, full_tokens=700).answer(TABLE, "Cell text trimmed, no rows dropped?")
    assert trimmed["rows_kept"] == 4 and trimmed["tokens_before_truncation"] == 700
    assert trimmed["truncated"] is True
    empty = _pipeline([], 0).answer(TABLE, "Nothing selected?")
    assert empty["cells"] == [] and empty["answer"] == "" and empty["numeric_answer"] is None
    with pytest.raises(RuntimeError, match="outside the 4x3 table"):
        _pipeline([(4, 0)], 0).answer(TABLE, "q")


def test_number_parsing_and_numeric_answer_helper():
    assert parse_number(" 1,846,513 ") == 1_846_513.0
    assert parse_number("-3.5") == -3.5 and parse_number("12%") == 12.0
    assert parse_number("1,23") is None and parse_number("NCR") is None and parse_number("") is None
    assert compute_numeric_answer(["1", "2"], "SUM") == (3.0, [])
    assert compute_numeric_answer(["1", "x"], "AVERAGE") == (None, ["x"])
    assert compute_numeric_answer([], "SUM") == (None, [])
    assert compute_numeric_answer(["a", "b"], "COUNT") == (2.0, [])
    assert compute_numeric_answer(["a"], "NONE") == (None, [])


def test_denotation_match_and_accuracy():
    lookup = _pipeline([(1, 0)], 0).answer(TABLE, "q")
    assert denotation_match(lookup, "cebu") and denotation_match(lookup, ["Cebu"])
    assert not denotation_match(lookup, "Davao")
    total = _pipeline([(0, 1), (2, 1)], 1).answer(TABLE, "q")
    assert denotation_match(total, 3_623_462) and denotation_match(total, "3,623,462")
    assert not denotation_match(total, 3_623_463)
    single = _pipeline([(3, 1)], 0).answer(TABLE, "q")
    assert denotation_match(single, 366358.0)  # one numeric cell under NONE compares as a number
    with pytest.raises(TypeError):
        denotation_match(lookup, True)
    assert denotation_accuracy([lookup, total], ["Cebu", 0]) == 0.5
    with pytest.raises(ValueError):
        denotation_accuracy([lookup], ["a", "b"])
