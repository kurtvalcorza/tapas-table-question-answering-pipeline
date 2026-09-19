# TAPAS Table Question Answering Pipeline

DIMER inference and bounded fine-tuning wrapper for **`google/tapas-large-finetuned-wtq`** — table question answering by cell selection plus one aggregation operator (NONE/SUM/AVERAGE/COUNT) — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot.

## Upstream alignment

- Model: `google/tapas-large-finetuned-wtq` (TAPAS large, 24 layers, hidden 1024, BERT-style encoder with cell-selection and aggregation heads, fine-tuned on SQA → WikiSQL → WikiTableQuestions; `reset` position embeddings)
- Revision: `f58317ab2577d17647d9acafa790c744a0388b30`
- Upstream weight license: Apache-2.0
- Upstream task: table question answering (WTQ); the model selects cells and an operator, it never emits a number
- Repository adaptation: a bounded fine-tuning contract (`evaluate`, `adapt`, `save_artifact`, `from_artifact`) over the last encoder blocks and the three heads, with the checkpoint's own weak-supervision loss; the base weights are never modified on disk. The numeric value for SUM/AVERAGE/COUNT is computed by this package from the selected cells and labelled as such.

## Quick start

```python
from tapas_table_qa_pipeline import TAPASTableQAPipeline

table = {
    "City": ["Manila", "Cebu", "Davao", "Baguio"],
    "Population (2020)": ["1,846,513", "964,169", "1,776,949", "366,358"],
    "Region": ["NCR", "Region VII", "Region XI", "CAR"],
}
pipe = TAPASTableQAPipeline.from_pretrained()          # cuda:0 if available, else cpu
result = pipe.answer(table, "What is the total population of Manila and Davao?")
print(result["aggregation"], result["cells"], result["numeric_answer"])   # SUM ['1,846,513', '1,776,949'] 3623462.0
```

`answer(table, query)` takes one table as `{column: [cells]}` or `[{column: cell}, ...]` — every header and cell a `str` — and one question, and returns `cells`, `coordinates`, `aggregation`, the upstream-style `answer` string, `numeric_answer` (pipeline arithmetic over the selected cells; `None` when a cell does not parse, listed in `unparsed_cells`), the four `aggregation_logits`, `n_tokens`, `tokens_before_truncation`, `rows_kept` and `truncated`. Cells are selected at a fixed sigmoid threshold `CELL_THRESHOLD = 0.5` and the operator by argmax, exactly as the upstream `convert_logits_to_predictions`. Ceilings: `MAX_ROWS = 64`, `MAX_COLUMNS = 32` (rejected), `MAX_TOKENS = 512` (the tokenizer truncates: cell text trimmed, then trailing rows dropped — reported, never silent), `MAX_QUERY_CHARS = 500`, `MAX_CELL_CHARS = 200`. `denotation_accuracy(results, golds)` is the shipped metric helper.

## Adaptation contract

```python
from tapas_table_qa_pipeline import (
    TAPASTableQAPipeline, build_sample_dataset, fetch_corpus, load_byod_dataset, read_corpus,
    split_dataset, wikisql_candidates,
)

splits = build_sample_dataset(wikisql_candidates(read_corpus(fetch_corpus())), seed=42)  # WikiSQL, 240 / 90 / 150
# or: splits = split_dataset(load_byod_dataset("my_questions.jsonl"), seed=42)          # {id, table, question, answer}

pipe = TAPASTableQAPipeline.from_pretrained()                       # cuda:0 if available, else cpu
frozen = pipe.evaluate(splits["test"])                              # accuracy, aggregation_accuracy, cell_accuracy, per_category
result = pipe.adapt(splits["train"], splits["validation"], epochs=4, lr=5e-5, trainable_layers=2)
adapted = pipe.evaluate(splits["test"])
pipe.save_artifact("outputs/adapter")                               # adapter.safetensors + manifest.json
again = TAPASTableQAPipeline.from_artifact("outputs/adapter")
```

- Records are `{id, table, question, answer, category?}`: `table` is `{column: [cells]}` with every header and cell a `str` (the `answer` ceilings), `answer` = `{aggregation: NONE|SUM|AVERAGE|COUNT, coordinates: [[row, col], ...], denotation: [cell, ...] | number}` — a `NONE` denotation is the selected cells, an operator denotation is the number the operator produces from them, and `validate_dataset` checks both (8..20,000 records, unique ids). `split_dataset` is a seeded split by normalised table content; `check_split_disjoint` asserts no table is shared.
- The default sample (`samples.py`) is the WikiSQL validation shard (`Salesforce/wikisql`, BSD-3-Clause) converted to parquet by the Hub at an immutable revision: fetched whole (3.6 MB), refused on any byte-count or SHA-256 mismatch, its SQL programmes executed in pure Python for the gold cells and value (`execute_sql`; `MAX`/`MIN` become a `NONE` lookup of the extreme cell, `AVG` becomes `AVERAGE`, the original operator is the record's `category`), and a seeded, table-disjoint draw of 80 lookups + 32 per operator for training and 15 / 25 per operator for validation / test. The shard is cached git-ignored under `weights/wikisql/`.
- `evaluate(records)` answers every record through `answer` and returns `denotation_metrics` (`metrics.py`): denotation accuracy (the WTQ criterion), aggregation accuracy (operator equals the record's) and cell accuracy (coordinates equal the record's as a set), overall, per category and per gold operator, plus `predictions`, `verdict` (`measured` / `measured-small-sample`), `truncated` and `adapted`. `first_cell_baseline` and `keyword_lookup_baseline` are the two non-neural references the tutorial scores beside the model.
- `adapt(train, val=None, *, epochs=4, lr=5e-5, batch_size=8, trainable_layers=2, seed=0, progress=None)` trains only the last `trainable_layers` encoder blocks and the cell-selection, column and aggregation heads (25,198,598 of 336,734,214 parameters by default) with the model's own weak-supervision loss — a lookup's gold cells label its tokens, an operator question carries only its number as `float_answer` — AdamW, gradient clipping at 1.0, seeded shuffling, batches padded to their longest table; epoch 0 records the frozen model and the epoch with the highest validation score (the mean of the three measures) is kept. The update is transactional: an exception restores the frozen weights. A record whose gold cells lie past the `MAX_TOKENS` truncation point is refused.
- `save_artifact(dir)` writes the trained tensors as `adapter.safetensors` plus a `manifest.json` (format `org.valcorza.tapas-large-wtq.adapter.v1`: base id, revision and weight digest, tensor names, file size and SHA-256, training configuration, epoch history); `from_artifact(dir)` re-verifies the base snapshot, checks the manifest, the digest and the exact tensor set before deserialising, refuses any tensor outside the encoder blocks and heads, and overlays the tensors onto a freshly loaded base.

## Weights layout

```
weights/tapas-large-wtq/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + sha256 (verified on every load)
  config.json                # TapasForQuestionAnswering architecture, aggregation_labels, max_num_rows/columns
  tokenizer_config.json, special_tokens_map.json, vocab.txt
  model.safetensors          # 1346985282 bytes, git-ignored
  README.md                  # upstream card listed in the manifest; not used by the loader
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` and refuses to load if any manifest file is missing or its SHA-256 differs. On a fresh clone (manifest committed, weights git-ignored) `from_pretrained(allow_download=True)` fetches only the missing files at the pinned revision; the default is to refuse. Without any manifest, `allow_download=True` loads from the Hub with `revision=f58317ab2577d17647d9acafa790c744a0388b30`.

## Tests and smoke

```
pip install -e . --no-deps
pytest -q -o addopts= tests      # offline, no weights needed; test_model_backed.py runs only with torch and the staged snapshot
```

Smoke (loads the verified snapshot on CPU; measured numbers are in `MODEL_CARD.md` → Runtime):

```python
from tapas_table_qa_pipeline import TAPASTableQAPipeline

pipe = TAPASTableQAPipeline.from_pretrained(device="cpu")
print(pipe.answer({"City": ["Manila", "Cebu"], "Region": ["NCR", "Region VII"]}, "Which city is in Region VII?")["cells"])
```

Runtime note: the `TapasTokenizer` in `transformers==4.57.6` predates `pandas==3.0.5` (pyarrow-backed string columns reject its `Cell` objects, and positional `row[col_index]` on a `Series` was removed); the loader builds the frame with `dtype=object` through a `DataFrame` subclass whose `iterrows()` yields position-indexed rows. No tokenizer code is modified.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tapas-table-question-answering-pipeline/blob/main/tutorials/tapas_table_qa_colab.ipynb)

`tutorials/tapas_table_qa_colab.ipynb` is declared `E2E` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the package's three modules (`pipeline.py`, `metrics.py`, `samples.py`), the model identity, the 6-file manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`; see `tutorials/README.md`). It stages and digest-verifies the snapshot, fetches the digest-pinned WikiSQL validation shard and executes its SQL programmes for gold answers, draws a table-disjoint sample (240 / 90 / 150 questions, stratified by operator), answers the synthetic city table through the inference contract with an input manifest and a rejection probe, scores the frozen model's denotation, aggregation and cell accuracy per question type on the 150 held-out questions beside the first-cell and keyword-lookup baselines, runs a bounded fine-tuning of the last two encoder blocks and the three heads with the model's own weak-supervision loss and validation-score epoch selection, re-scores the held-out split and the city table, exports the adapter and reloads it with verified parity, and writes `tapas_table_qa_train.jsonl`, `tapas_table_qa_input_manifest.json`, `tapas_table_qa_evaluation_report.json`, `tapas_table_qa_answers.csv`, `tapas_table_qa_adapter/` and `tapas_table_qa_result.json` under `outputs/`.

The default path runs on CPU and uses CUDA automatically when present (about 31 minutes on the build workstation's CPU after the downloads, a multiple of that on a 2-vCPU hosted runtime; about 2.5 minutes on an RTX 5070 Ti). The metrics it prints are one seeded split of one 480-question sample — evidence that the adaptation contract works, not an accuracy benchmark or production-fitness evidence.

## Release status

**Candidate.** Static/unit checks do not constitute clean-runtime notebook evidence. The clean-runtime run of the `E2E` standalone tutorial is pending; complete `docs/release-verification.md` against the exact release revision before calling the notebook release-grade. See `STATUS.md`.

## Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — MODEL_CARD_SPEC 1.1 card
- [`docs/WEIGHTS.md`](docs/WEIGHTS.md) — weight provenance and hosting
- [`STATUS.md`](STATUS.md) — release status

## Licensing

Repository code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
