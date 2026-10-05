"""Regression tests for the 2026-10-05 notebook review findings (TPQ-M1..M5, TPQ-m1).

Every test needs only CI's dependencies and no model: the notebook's own cell sources are executed with stand-ins
where a model would be needed, and restore_base() is exercised on a stand-in model, not the checkpoint. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tapas_table_qa_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "tapas_table_qa_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PIPELINE = ROOT / "src" / "tapas_table_qa_pipeline" / "pipeline.py"
STEM = "tapas_table_qa"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- TPQ-M1: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------


def test_tpq_m1_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source


def test_tpq_m1_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    spec = importlib.util.spec_from_file_location("_review_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_lock(build._pins(ROOT), lock_text)


def test_tpq_m1_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker — with every variable later cells created — is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- TPQ-M2: every adaptation starts from the pinned base -------------------------------------------------------


def test_tpq_m2_adapt_and_load_artifact_restore_the_base_first():
    """Torch-backed, so the order inside adapt is checked statically: pre-call state kept, base restored, then epoch 0."""
    text = PIPELINE.read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def save_artifact(")]
    order = [adapt.index(m) for m in ("previous_state = {", "restored = self.restore_base()", "self._remember_base(names)", "frozen_state = {", '"note": "frozen model"', "for epoch in range(1, epochs + 1):")]
    assert order == sorted(order)
    failure = adapt[adapt.index("except BaseException:") :]
    assert failure.index("model.load_state_dict(previous_state, strict=False)") < failure.index("self.adapter = previous_adapter") < failure.index("raise")
    assert '"started_from": "pinned base"' in adapt
    load = text[text.index("    def load_artifact(") : text.index("    def from_artifact(")]
    assert load.index("self.restore_base()") < load.index("self._remember_base(sorted(tensors))") < load.index("model.load_state_dict(")


class _Tensor:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)

    def detach(self):
        return self

    def clone(self):
        return _Tensor(self.value.copy())


class _Model:
    def __init__(self):
        self.state = {"encoder.layer.23.w": _Tensor([1.0, 2.0]), "aggregation_classifier.w": _Tensor([3.0]), "embeddings.w": _Tensor([4.0])}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, values, strict=True):
        assert strict is False
        for name, tensor in values.items():
            self.state[name] = _Tensor(tensor.value.copy())

    def eval(self):
        return self


def test_tpq_m2_restore_base_undoes_every_earlier_change_stand_in():
    """restore_base() and _remember_base() on a stand-in model with the state_dict interface (numpy, not torch)."""
    from tapas_table_qa_pipeline import TAPASTableQAPipeline

    model = _Model()
    pipe = TAPASTableQAPipeline(_runner=lambda *a: {}, _model=model, _tokenizer=object(), _frame_cls=object())
    assert pipe.restore_base() == []
    pipe._remember_base(["encoder.layer.23.w", "aggregation_classifier.w"])
    model.state["encoder.layer.23.w"] = _Tensor([9.0, 9.0])
    model.state["aggregation_classifier.w"] = _Tensor([9.0])
    pipe._remember_base(["encoder.layer.23.w"])  # a second run keeps the first (base) value
    pipe.adapter = {"best_epoch": 2}
    assert pipe.restore_base() == ["aggregation_classifier.w", "encoder.layer.23.w"]
    assert model.state["encoder.layer.23.w"].value.tolist() == [1.0, 2.0] and model.state["aggregation_classifier.w"].value.tolist() == [3.0]
    assert model.state["embeddings.w"].value.tolist() == [4.0] and pipe.adapter is None


def test_tpq_m2_byod_rerun_restores_the_base_and_the_experiment_has_its_own_pipeline(notebook):
    section_4 = _cell(notebook, "USE_BYOD = False")
    assert section_4.index("restored_tensors = pipe.restore_base()") < section_4.index("if USE_BYOD:")
    experiment = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "experiment_pipe = TAPASTableQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)" in experiment
    assert f"Path('outputs/{STEM}_experiment')" in experiment
    assert "raise RuntimeError(f'the experiment changed a default export: {unchanged}')" in experiment
    assert "'epoch_0_val_score'" in experiment
    assert not re.search(r"(?<!experiment_)pipe\.adapt\(", experiment)
    assert "**Predict → Change one thing → Run → Observe → Explain**" in _markdown(notebook)
    assert "they do not affect the default path" not in _markdown(notebook)


# --- TPQ-M3: quality outcomes are reported verdicts -------------------------------------------------------------


def test_tpq_m3_no_quality_assert_remains(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = re.findall(r"(?m)^\s*assert .*$", code)
    assert asserts == ["assert parity['identical_answers'] == parity['of']"]


CATS = ("average", "count", "lookup", "max", "min", "sum")


def _scores(acc, agg, cells=0.8, per=None):
    per = per or {}
    return {
        "accuracy": acc, "aggregation_accuracy": agg, "cell_accuracy": cells, "n": 150, "verdict": "measured", "truncated": 0, "definitions": {},
        "per_category": {c: {"n": 25, "accuracy": acc, "aggregation_accuracy": per.get(c, agg), "cell_accuracy": cells} for c in CATS},
        "baseline": "stand-in",
    }


def test_tpq_m3_negative_results_are_recorded_and_do_not_stop_the_notebook(notebook, tmp_path, monkeypatch):
    """Sections 6 and 8 with stand-ins: the keyword lookup beats the frozen model, and aggregation accuracy cannot rise
    (the frozen model already picks every operator). Both cells complete and record the verdicts (no model)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scores = iter([_scores(0.5, 1.0), _scores(0.45, 1.0, per={"sum": 0.5}), _scores(0.5, 1.0)])

    class StandIn:
        def evaluate(self, records):
            return next(scores)

    ns = {
        "pipe": StandIn(), "test_records": [], "val_records": [], "time": __import__("time"), "json": json,
        "first_cell_baseline": lambda r: _scores(0.1, 0.5), "keyword_lookup_baseline": lambda r: _scores(0.6, 0.7),
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "MODEL_KEY": "stand-in", "data_source": "stand-in", "dataset_manifests": {"test": {"digest": "d"}},
        "disjoint": {}, "adapt_result": {"history": [], "trainable_names": []}, "adapt_seconds": 0.0,
    }
    exec(_cell(notebook, "baseline_first = first_cell_baseline("), ns)
    assert ns["frozen_verdict"]["accuracy"] == "a baseline ties or beats the frozen model"
    exec(_cell(notebook, "adapted_test = pipe.evaluate(test_records)"), ns)
    comparison = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))["comparison"]
    assert comparison["verdicts"]["adapted_vs_frozen_aggregation_accuracy"] == "no gain"
    assert comparison["verdicts"]["adapted_vs_frozen_accuracy"] == "worse"
    assert comparison["operator_regressions"] == ["sum"]


# --- TPQ-M4: the held-out story matches the release run -------------------------------------------------------


def test_tpq_m4_prose_names_its_runs_and_discusses_the_sum_regression(notebook):
    markdown = _markdown(notebook)
    for stale in ("with the lookups untouched", "lookups stayed at 0.92", "`average`, `min` and `sum` each gained one question", "the operator choice is what these questions teach"):
        assert stale not in markdown, stale
    assert "Kaggle T4 release run" in markdown and "build record" in markdown
    assert "0.88 to 0.56" in markdown and "0.82 → 0.813" in markdown
    assert "unlearned" in markdown


# --- TPQ-M5: guided layer and infrastructure labelling ----------------------------------------------------------


def test_tpq_m5_guided_layer_is_present(notebook):
    markdown = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "## 10. Change one thing", "**Learner:**"):
        assert heading in markdown, heading
    assert markdown.count("**Predict") >= 7
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 7
    assert markdown.count("**What to notice:**") >= 6


def test_tpq_m5_infrastructure_cells_are_labelled_and_collapsed(notebook):
    infra = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") == "form"]
    assert len([c for c in infra if c["metadata"].get("dimer", {}).get("embedded_module")]) == 3
    titled = [c["source"].splitlines()[0] for c in infra if not c["metadata"].get("dimer")]
    assert len(titled) == 3 and all(t.startswith("# @title Infrastructure:") for t in titled), titled


def test_tpq_m5_no_template_placeholders_leak(notebook):
    learner = "\n".join(c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in learner, leftover
    assert "}}" not in _markdown(notebook)


# --- TPQ-m1: BYOD contract --------------------------------------------------------------------------------------


def _record(i: int, j: int = 0) -> dict:
    return {"id": f"q{i}_{j}", "table": {"Name": [f"n{i}a", f"n{i}b"], "Value": ["1", str(i + 2)]}, "question": f"Which name is first in table {i}?", "answer": {"aggregation": "NONE", "coordinates": [[0, 0]], "denotation": [f"n{i}a"]}, "category": "lookup"}


def _jsonl(path: Path, tables: int, per_table: int = 1, *, bom: bool = False, broken_line: int | None = None) -> Path:
    lines = [json.dumps(_record(i, j)) for i in range(tables) for j in range(per_table)]
    if broken_line is not None:
        lines[broken_line - 1] = lines[broken_line - 1][:-3]
    path.write_text(("﻿" if bom else "") + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_tpq_m1_stated_minimum_is_what_the_split_accepts(tmp_path, notebook):
    need = sm.min_byod_records()
    assert need["total"] == 12 and need["tables"] == 12
    for tables, per_table in ((12, 1), (6, 2)):
        split = sm.split_dataset(sm.load_byod_dataset(_jsonl(tmp_path / f"ok{tables}.jsonl", tables, per_table)), seed=42)
        assert len(split["train"]) >= 8 and split["validation"] and split["test"]
    with pytest.raises(ValueError, match=r"the split leaves 7 training questions.*supply at least 12 questions"):
        sm.split_dataset(sm.load_byod_dataset(_jsonl(tmp_path / "small.jsonl", 11)), seed=42)
    assert "**12 questions" in _markdown(notebook)


def test_tpq_m1_bom_and_broken_lines_are_handled_with_file_line_numbers(tmp_path):
    assert len(sm.load_byod_dataset(_jsonl(tmp_path / "bom.jsonl", 12, bom=True))) == 12
    with pytest.raises(ValueError, match=r"broken.jsonl line 5: not valid JSON"):
        sm.load_byod_dataset(_jsonl(tmp_path / "broken.jsonl", 12, broken_line=5))
    (tmp_path / "empty.jsonl").write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="holds no records"):
        sm.load_byod_dataset(tmp_path / "empty.jsonl")


def _section_4(notebook: dict, path: str) -> str:
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)


def _section_4_namespace(restored: list) -> dict:
    from tapas_table_qa_pipeline import metrics as mt
    from tapas_table_qa_pipeline import pipeline as pl

    ns = {}
    for module in (pl, mt, sm):
        ns.update({k: getattr(module, k) for k in dir(module) if not k.startswith("__")})
    pipe = types.SimpleNamespace(adapter={"best_epoch": 2}, restore_base=lambda: restored.append(True) or ["a"])
    ns.update({"os": __import__("os"), "Path": Path, "pipe": pipe, "__name__": "__main__"})
    return ns


def test_tpq_m1_byod_path_runs_section_4_outside_colab_from_the_base(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _jsonl(tmp_path / "mine.jsonl", 14)
    restored: list = []
    ns = _section_4_namespace(restored)
    exec(_section_4(notebook, "mine.jsonl"), ns)
    out = capsys.readouterr().out
    assert restored == [True], "a BYOD re-run must put the model back to the pinned base first"
    assert ns["raw_rows"] == {"byod": 14, "effective_minimum": 12}
    assert "held-out test questions" in out
    assert (tmp_path / "outputs" / f"{STEM}_train.jsonl").is_file()


def test_tpq_m1_upload_outside_colab_cancelled_and_bad_path_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(_section_4(notebook, ""), _section_4_namespace([]))
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nowhere.jsonl' is not a file"):
        exec(_section_4(notebook, "nowhere.jsonl"), _section_4_namespace([]))
    for uploaded, message in (({}, "received 0"), ({"a.jsonl": b"", "b.jsonl": b""}, "received 2")):
        google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files, google.colab = files, colab
        for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, module)
        with pytest.raises(ValueError, match=message):
            exec(_section_4(notebook, ""), _section_4_namespace([]))
