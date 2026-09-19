# Release verification

`tutorials/tapas_table_qa_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are
**not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 6-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned WikiSQL
  parquet revision is the one other 40-hex string allowed);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `TAPASTableQAPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `wikisql_candidates`, `build_sample_dataset(candidates, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`,
  `validate_dataset` per split, `check_split_disjoint`, `write_dataset_jsonl`, the four dataset refusal probes, the
  ceiling print, `validate_inputs` with the numeric-cell refusal probe, `answer` with the sanity checks and the
  per-grid `evaluation_report` on the city table, `first_cell_baseline`, `keyword_lookup_baseline`, `pipe.evaluate`
  on the frozen model with the baseline assertion, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on
  the validation and test splits after adaptation with the aggregation-accuracy assertion, `answer` +
  `evaluation_report` on the city table after adaptation, the answers CSV, `pipe.save_artifact`,
  `TAPASTableQAPipeline.from_artifact` and the reload-parity assertion, and the result fields `decision_rule`,
  `numeric_answer_source`, `weight_file` / `weight_format` / `weight_sha256` and the `corpus` block), the six
  expected `outputs/` paths, the learner-facing statements (Apache-2.0 weights, the pipeline-computed numeric answer,
  no abstention, adaptation with labelled questions, the BSD-3-Clause corpus, the weak-supervision loss, the three
  measures, the two non-neural baselines, no dispersion estimate, the leakage and supervision guidance, the SQA
  exclusion, the snapshot note) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any
  `git clone` / `github.com/kurtvalcorza` / repository import on the primary path, a mutable `revision='main'`,
  direct `from transformers import` / `TapasTokenizer` / `TapasForQuestionAnswering` / `convert_logits_to_predictions`
  / `DataFrame(` / `sigmoid(` / `from huggingface_hub import` / `urllib.request` / `pyarrow` / `safetensors` imports /
  `torch.optim` / `.backward(` / `requires_grad` / `pipe._model` / `extractall(` use **outside the carried module
  cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs `pytest`, `ruff` and `numpy`, the package with `--no-deps`, runs `ruff check src tests tools`,
`tools/build_notebook.py --check`, and the offline unit suite (`tests/`, including `test_adaptation.py`,
`test_role_helpers.py`, `test_import_boundary.py`, `test_notebook_parity.py`; injected runner and shard fetcher, no
weights, no `torch` — `tests/test_model_backed.py` is skipped without `torch` or the snapshot). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/tapas-large-wtq/` or the shard cache `weights/wikisql/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub and fetches the pinned shard, so neither directory may be
   seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 4`, `LEARNING_RATE = 5e-5`, `BATCH_SIZE = 8`,
   `TRAINABLE_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `safetensors==0.8.0`,
   `numpy==2.5.3`, `pandas==3.0.5`, `pyarrow==25.0.1`, `huggingface-hub==0.36.2` (an interpreter restart after the
   install is expected where the runtime's preinstalled torch, numpy or pandas differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `TAPASTableQAPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `denotation_accuracy`, `denotation_metrics`,
     `first_cell_baseline`, `keyword_lookup_baseline`, `fetch_corpus`, `read_corpus`, `execute_sql`,
     `wikisql_candidates`, `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_jsonl` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 6 manifest entries fetched from `google/tapas-large-finetuned-wtq` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (6 files, the 1.35 GB
     `model.safetensors` re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified
     directory with `source` `local-snapshot`;
   - Section 4: `fetch_corpus` fetching the 3,630,670-byte shard with the pinned SHA-256; 8,421 questions over
     2,630 tables decoded, 4,972 candidates, the seeded table-disjoint draw 240 / 90 / 150 (80 lookups + 32 per
     operator in training, 15 / 25 per operator elsewhere) with `check_split_disjoint` reporting no shared table,
     the category and operator counts and the three dataset digests printed; `outputs/…_train.jsonl` written; the
     four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MAX_ROWS` 64, `MAX_COLUMNS` 32, `MAX_TOKENS` 512, `MAX_QUERY_CHARS` 500,
     `MAX_CELL_CHARS` 200, `AGGREGATIONS`, `CELL_THRESHOLD` 0.5, `MIN_RECORDS` 8, `MAX_RECORDS` 20000) surfaced; the
     city table authored; `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one
     recorded rejection finding from the numeric-cell probe); the three questions answered with every sanity check
     `True` (`Cebu` NONE, `COUNT` → 4.0, `SUM` → 3,623,462.0 in every recorded run) and the per-grid
     `evaluation_report` verdict `sample-sanity`;
   - Section 6: the first-cell floor (accuracy 0.073), the keyword lookup (0.367) and the frozen model's test score
     (denotation 0.82 / aggregation 0.567 / cell 0.82 in the RTX 5070 Ti build record; lookups 0.92, `count` 0.64)
     with the per-type breakdown, and the cell's assertion that the frozen model is above both baselines;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 25,198,598 trainable of 336,734,214 parameters,
     and a four-epoch history with the validation score selecting the epoch (`best_epoch` 2 in the build record;
     validation aggregation accuracy 0.567 → 0.82 after the first epoch);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on the three
     measures, the per-type breakdown and `outputs/…_evaluation_report.json` written (the cell asserts the adapted
     aggregation accuracy exceeds the frozen one — 0.807 versus 0.567 in the build record, with denotation accuracy
     0.82 → 0.84 and cell accuracy 0.82 → 0.847; `average`, `min` and `sum` +0.04 each, `count`, `max` and lookups
     unchanged);
   - Section 9: the city table re-answered by the adapted model with the `sample-sanity` report and
     `outputs/…_answers.csv` written; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors,manifest.json}` (38 tensors, 100,798,688 bytes) and
     `TAPASTableQAPipeline.from_artifact` reloading it with 8/8 identical answers on eight test questions (the
     cell asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the
     snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus` block, the inference-contract
     reports, the comparison, the artifact digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the shard cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `tapas_table_qa_colab.ipynb` (`E2E`) | pending | — | — | **PENDING** — a clean-runtime execution of the `E2E` blob has not been recorded yet |
| `tapas_table_qa_colab.ipynb` (`TASK-INFERENCE`, superseded) | `8ef61d5` / `08f1778d5ce6` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-tapas-table-qa` v1) | PASS — 8/8 code cells (1 restart after install cell), 258.7 s, 14 files, 1347 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/tapas_table_qa_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/tapas_table_qa_colab.ipynb`). Wall times, when recorded, are the sum of per-cell
times reported by the executor and include installs and the model download; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | generated at `b0ebfe4` / blob `11134191028d` | Local Windows-venv harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, pandas 3.0.5, snapshot and shard pre-staged; another job shared the CPU for part of the run | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (6 files), the shard re-hashed from the pre-staged cache, 4,972 candidates, split 240 / 90 / 150 by table, four refusal probes raised, the city table answered (`Cebu` / COUNT 4.0 / SUM 3,623,462.0), baselines 0.073 / 0.367, frozen test denotation 0.82 / aggregation 0.567 / cell 0.82 in 75.6 s, four epochs 1,401.3 s (validation score 0.707 → 0.785 → 0.793 → 0.800 → 0.796, epoch 3 kept), adapted test 0.82 / 0.827 / 0.833 (`average` 0.88 → 0.92, `count` 0.64 → 0.60, other types unchanged), city table unchanged after adaptation, adapter 100,798,688 B / 38 tensors, reload parity 8/8, 6 outputs written; the committed blob differs from the executed one in markdown prose only (timing figures filled in after this run) | 1864.7 s | PASS — pre-flight only; not promotion evidence |
| 2026-09-20 | generated at `b0ebfe4` / blob `11134191028d` | Local WSL harness (same `run_nb_local.py`, `CUDA_VISIBLE_DEVICES=0`), Python 3.12.3, torch 2.14.0+cu130, RTX 5070 Ti (`cuda:0`), snapshot and shard pre-staged | Default sample path, all 11 code cells; frozen 0.82 / 0.567 / 0.82 in 9.7 s, four epochs 76.7 s, epoch 2 kept, adapted 0.84 / 0.807 / 0.847 (`average`, `min`, `sum` +0.04; `count`, `max`, lookups unchanged), reload parity 8/8 | 147.7 s | PASS — pre-flight only; not promotion evidence |

## Current status

**Candidate.** The `E2E` notebook's source passes static validation, the generator parity checks and the offline unit
suite; the model-backed regressions (`tests/test_model_backed.py`, 7 tests) pass on CPU and on the RTX 5070 Ti in
WSL; the local pre-flight above executed the default path top-to-bottom. None of that is REL1/REL10 supported-runtime
evidence: the registry stays **Candidate** until a clean-runtime run of the committed blob — no repository checkout,
an empty Hugging Face cache, no pre-staged snapshot or shard — is recorded in the tables above and an integrator
promotes it.

Facts a reviewer should still weigh: the frozen WTQ checkpoint is already a strong lookup model on WikiSQL tables
(denotation 0.82, lookups 0.92 in the build record), so the adaptation gain is the operator choice (aggregation
accuracy 0.567 → 0.807) with denotation accuracy moving by three questions of 150 (0.82 → 0.84), which is why the
notebook asserts aggregation accuracy and only reports denotation accuracy; the 90-question validation split moves
each measure in steps of one question, which is why the epoch is selected on the mean of the three; the build
record's other configurations — four trainable blocks, or a training draw with as many operator questions as
lookups — raised aggregation accuracy as much while costing lookups (0.92 → 0.84), and on one draw denotation
accuracy did not move at all; supervision for operator questions is the number only (no cell labels), because
labelling their cells too let the model's aggregate mask re-route them to cell selection; and the city table
re-answered after adaptation is three questions of evidence about behaviour outside the corpus, not a measurement.
