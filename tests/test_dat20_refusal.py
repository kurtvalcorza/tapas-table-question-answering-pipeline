"""DAT19/DAT20: a refused BYOD input renders exactly one `Refused: <file>: <reason>` line, no traceback.

Executes the committed Section 4 cell with the real loader, splitter and validator. Covers the scripted
hosted refusal input (`BYOD_PATH='/etc/hostname'`: a file that is not JSONL), a directory, an upload
(named by its own file name), the in-cell invalidation, and that a genuine error keeps its own type.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from test_notebook_review_fixes import NOTEBOOK, STEM, _section_4, _section_4_namespace

REFUSAL_CELL = "tapas_table_qa-19"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _refused(notebook, path: str, ns: dict | None = None) -> str:
    with pytest.raises(ValueError) as caught:
        exec(_section_4(notebook, path), ns or _section_4_namespace([]))
    assert caught.value.dimer_refusal is True
    rendered = caught.value._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0], rendered
    assert rendered[0].startswith("Refused: ") and "Traceback" not in rendered[0]
    return rendered[0]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    return tmp_path


def test_scripted_hosted_input_a_file_that_is_not_jsonl(notebook, workdir):
    (workdir / "hostname").write_text("colab-vm\n", encoding="utf-8")
    line = _refused(notebook, str(workdir / "hostname"))
    assert line.startswith("Refused: hostname: line 1: not valid JSON"), line


def test_a_directory(notebook, workdir):
    line = _refused(notebook, str(workdir))
    assert line.startswith(f"Refused: {workdir}: BYOD_PATH is not a file"), line


def test_an_upload_is_named_by_its_own_file_name(notebook, workdir, monkeypatch):
    google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
    files.upload = lambda: {"questions.jsonl": b"{broken\n"}
    colab.files, google.colab = files, colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, module)
    line = _refused(notebook, "")
    assert line.startswith("Refused: questions.jsonl: line 1: not valid JSON"), line


def test_refusal_invalidates_earlier_results_in_the_cell(notebook, workdir):
    (workdir / "outputs").mkdir()
    (workdir / "outputs" / f"{STEM}_evaluation_report.json").write_text("{}", encoding="utf-8")
    ns = _section_4_namespace([])
    ns.update(splits={"train": ["stale"]}, adapt_result={"stale": True})
    line = _refused(notebook, "nowhere.jsonl", ns)
    assert "Removed 1 earlier output(s)" in line and not list((workdir / "outputs").iterdir())
    assert "splits" not in ns and "adapt_result" not in ns


def test_genuine_error_keeps_its_type(notebook, workdir):
    (workdir / "ok.jsonl").write_text("{}\n", encoding="utf-8")
    ns = _section_4_namespace([])

    def broken(path):
        raise RuntimeError("internal bug")

    ns["load_byod_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        exec(_section_4(notebook, str(workdir / "ok.jsonl")), ns)
    assert not getattr(caught.value, "dimer_refusal", False)


def test_refusal_cell_id(notebook):
    ids = [c["id"] for c in notebook["cells"] if "def refuse_byod" in "".join(c["source"])]
    assert ids == [REFUSAL_CELL]
