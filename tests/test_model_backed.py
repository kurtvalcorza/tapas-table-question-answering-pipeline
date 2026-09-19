"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): evaluation with
the per-category breakdown, a one-epoch adaptation of the last encoder block on a dozen small tables, the
artifact round trip, the loader's scope check, the transactional guarantee and — where CUDA is visible —
the same path on the accelerator. Skipped when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from tapas_table_qa_pipeline import DEFAULT_WEIGHTS_DIR, WEIGHT_FILE, TAPASTableQAPipeline

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

CITIES = ["Manila", "Cebu", "Davao", "Baguio", "Iloilo", "Zamboanga"]


def _table(i):
    return {
        "City": [f"{c} {i}" for c in CITIES],
        "Population": [str(1000 + 137 * (i + k)) for k in range(6)],
        "Region": ["NCR", "VII", "XI", "CAR", "VI", "IX"],
    }


@pytest.fixture(scope="module")
def records():
    out = []
    for i in range(16):
        table = _table(i)
        kind = i % 4
        if kind == 0:
            out.append({"id": f"q{i:02d}", "table": table, "question": f"Which region is {table['City'][1]} in?", "answer": {"aggregation": "NONE", "coordinates": [[1, 2]], "denotation": ["VII"]}, "category": "lookup"})
        elif kind == 1:
            out.append({"id": f"q{i:02d}", "table": table, "question": "How many cities are in region NCR or CAR?", "answer": {"aggregation": "COUNT", "coordinates": [[0, 0], [3, 0]], "denotation": 2.0}, "category": "count"})
        elif kind == 2:
            value = float(int(table["Population"][0]) + int(table["Population"][2]))
            out.append({"id": f"q{i:02d}", "table": table, "question": f"What is the total population of {table['City'][0]} and {table['City'][2]}?", "answer": {"aggregation": "SUM", "coordinates": [[0, 1], [2, 1]], "denotation": value}, "category": "sum"})
        else:
            value = (int(table["Population"][1]) + int(table["Population"][3])) / 2
            out.append({"id": f"q{i:02d}", "table": table, "question": f"What is the average population of {table['City'][1]} and {table['City'][3]}?", "answer": {"aggregation": "AVERAGE", "coordinates": [[1, 1], [3, 1]], "denotation": value}, "category": "average"})
    return out


@pytest.fixture(scope="module")
def pipe():
    return TAPASTableQAPipeline.from_pretrained(device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_evaluate_scores_the_frozen_model_with_the_breakdown(pipe, records):
    metrics = pipe.evaluate(records[:8])
    assert metrics["n"] == 8 and 0.0 <= metrics["accuracy"] <= 1.0 and metrics["adapted"] is False
    assert set(metrics["per_category"]) == {"average", "count", "lookup", "sum"}
    assert metrics["verdict"] == "measured-small-sample" and metrics["truncated"] == 0
    assert pipe.weight_sha256 is not None and len(pipe.weight_sha256) == 64
    # evaluate() answers through the same path as answer()
    first = pipe.answer(records[0]["table"], records[0]["question"])
    assert metrics["predictions"][0]["aggregation"] == first["aggregation"]


def test_encode_for_training_labels_cells_for_none_and_only_the_number_for_operators(pipe, records):
    encoded, float_answer = pipe._encode_for_training(records[0])
    assert int(encoded["labels"].sum()) > 0 and float_answer != float_answer  # NaN for NONE
    encoded, float_answer = pipe._encode_for_training(records[1])
    assert int(encoded["labels"].sum()) == 0 and float_answer == 2.0
    assert set(encoded) >= {"input_ids", "attention_mask", "token_type_ids", "labels", "numeric_values", "numeric_values_scale"}


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], records[12:], epochs=1, trainable_layers=1, batch_size=4)
    assert result["n_trainable"] == 12_602_374  # one encoder block + the three heads
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert set(result["history"][1]["val"]) == {"accuracy", "aggregation_accuracy", "cell_accuracy", "n", "score"}
    assert all(n.startswith(("tapas.encoder.layer.23.", "output_", "column_output_", "aggregation_classifier")) for n in result["trainable_names"])
    assert not any(n.startswith(("tapas.embeddings", "tapas.pooler")) for n in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"]) and manifest["base"]["weight_sha256"] == pipe.weight_sha256
    reloaded = TAPASTableQAPipeline.from_artifact(artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    a = [pipe.answer(r["table"], r["question"]) for r in records[:4]]
    b = [reloaded.answer(r["table"], r["question"]) for r in records[:4]]
    assert all(x["cells"] == y["cells"] and x["aggregation_logits"] == y["aggregation_logits"] for x, y in zip(a, b, strict=True))
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], None, epochs=2, trainable_layers=1, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = TAPASTableQAPipeline.from_artifact(artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["trainable_layers"] == 1


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, records, tmp_path):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:12], None, epochs=1, trainable_layers=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        TAPASTableQAPipeline.from_artifact(fewer, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [{**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        TAPASTableQAPipeline.from_artifact(extra, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    (other_layers / "manifest.json").write_text(json.dumps({**manifest, "adapter": {**manifest["adapter"], "trainable_layers": 2}}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        TAPASTableQAPipeline.from_artifact(other_layers, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:12], None, epochs=2, trainable_layers=1, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_evaluate_adapt_and_reload_run_on_a_cuda_device(records, tmp_path):
    cuda = TAPASTableQAPipeline.from_pretrained(device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    assert cuda.device == "cuda:0"
    metrics = cuda.evaluate(records[:8])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    result = cuda.adapt(records[:12], records[12:], epochs=1, trainable_layers=1, batch_size=4)
    assert result["best_epoch"] in (0, 1) and result["history"][1]["train_loss"] > 0.0
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = TAPASTableQAPipeline.from_artifact(artifact, device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    a = [cuda.answer(r["table"], r["question"]) for r in records[:4]]
    b = [reloaded.answer(r["table"], r["question"]) for r in records[:4]]
    assert all(x["cells"] == y["cells"] for x, y in zip(a, b, strict=True))
