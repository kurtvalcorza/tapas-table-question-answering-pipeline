# Release verification

`tutorials/tapas_table_qa_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until the exact notebook revision has
executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime
evidence under DIMER Notebook Specification 1.1. This file is the durable release-gate record for
the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary
  path; exactly one cell tagged `embedded_module` equal to `src/tapas_table_qa_pipeline/pipeline.py` after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 6-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py`
  output; the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `TAPASTableQAPipeline.from_pretrained(weights_dir=...)`, `validate_inputs(table, queries, names=query_ids)`,
  `pipe.answer(table, query)`, `evaluation_report(results, golds, sample_kind=sample_kind)`), the ceiling print, the
  non-string-cell rejection probe, the aggregation/coordinate/COUNT sanity checks, the answers CSV header, the
  exported decision rule and `numeric_answer_source`, the four exports, the learner-facing statements (no adaptation,
  the model never emits a number, the numeric answer is the pipeline's arithmetic, fixed sigmoid threshold + argmax,
  `denotation_accuracy` as the WTQ criterion, `sample-sanity` with golds / `not-measurable` without, tokenizer
  truncation, strings-only cells, SQA exclusion, the `answer` contract) and the gated-off BYOD default listed in the
  validator; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary
  path, a mutable `revision='main'`, direct `transformers`, `huggingface_hub` or `pandas.DataFrame` calls **outside
  the carried module cell**, `TapasTokenizer`, `TapasForQuestionAnswering`, `convert_logits_to_predictions(`,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also runs `ruff`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`,
`tests/test_notebook_parity.py`; injected runner, no weights, no pandas). These are source/provenance and unit
checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 either way) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim, cell by cell, in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or pandas that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install (`pandas==3.0.5` is a likely conflict on hosted kernels) |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, empty model cache, no pre-staged `model.safetensors` under `weights/tapas-large-wtq/` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container
   executor above) with **no repository checkout**, an empty Hugging Face
   cache, and no pre-staged `model.safetensors` under `weights/tapas-large-wtq/`;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`; `BYOD_QUESTIONS` is unused on that path);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= the
   `pyproject.toml` pins `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `tokenizers==0.22.2`, `huggingface-hub==0.36.2`,
   `safetensors==0.8.0`, `numpy==2.5.3`, `pandas==3.0.5`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executing (defining `TAPASTableQAPipeline`, `validate_inputs`, `evaluation_report`,
     `denotation_accuracy` and the ceilings) with no import of the repository package;
   - the synthetic 4×3 table and three questions authored in code with the sample SHA-256 printed;
   - the inline `MANIFEST` asserted against the module identity and written to `weights/tapas-large-wtq/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reporting `['model.safetensors']` fetched from
     `google/tapas-large-finetuned-wtq` at the immutable revision, `verify_snapshot` returning the
     6-entry manifest, and `TAPASTableQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified
     directory with `source 'local-snapshot'`;
   - ceilings `MAX_ROWS = 64`, `MAX_COLUMNS = 32`, `MAX_TOKENS = 512`, `MAX_QUERY_CHARS = 500`,
     `MAX_CELL_CHARS = 200`, `AGGREGATIONS = ('NONE', 'SUM', 'AVERAGE', 'COUNT')`, `CELL_THRESHOLD = 0.5`
     printed, and `validate_inputs` writing `outputs/tapas_table_qa_input_manifest.json` (verdict `accepted`, table
     `n_rows 4`, `n_columns 3`, `numeric_cells 4`, three query entries, one recorded rejection finding from the
     numeric-cell probe) before model execution;
   - `answer` returning, for each of the three questions, an aggregation in `AGGREGATIONS`, coordinates inside the
     table, cells equal to the coordinates' contents, `rows_kept 4`, `truncated False`, all seven sanity checks true;
     the model card's smoke observed `Cebu` (NONE), `COUNT` over the four city cells → 4.0 and `SUM` over
     `1,846,513` + `1,776,949` → 3,623,462.0 (observations, not expected values);
   - `evaluation_report` writing `outputs/tapas_table_qa_evaluation_report.json` with verdict `sample-sanity`, one
     `denotation_accuracy` metric entry with `n 3` and the single-sample estimation note, and the plumbing-check
     line printed;
   - `outputs/tapas_table_qa_answers.csv` written with one identified row per question and
     `outputs/tapas_table_qa_result.json` written with the table, the results, the decision rule,
     `numeric_answer_source`, `NOTEBOOK_SOURCE`, model identifier, immutable model revision, model licence,
     snapshot summary, runtime versions (including `pandas`), device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, pandas, NumPy,
   device), model identifier and immutable revision, whether the model cache and weights directory
   were clean, outcome, produced outputs, the three answers with their aggregations and numeric values (as
   observations, not a metric), the sample-sanity `denotation_accuracy` value, and any warning or applicable
   `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `tapas_table_qa_colab.ipynb` | `8ef61d5` / `08f1778d5ce6` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-tapas-table-qa` v1) | PASS — 8/8 ok (1 restart after install cell) |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/tapas_table_qa_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/tapas_table_qa_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include installs and the model download; they are measurements for the stated
runtime, not general estimates.

No execution of the notebook has been recorded. The only runtime measurements that exist for this
repository are the pipeline smoke run documented in `MODEL_CARD.md` (Windows venv, CPU float32,
`HF_HUB_OFFLINE=1`: load and verify 6.83 s; the three questions on the same 4×3 table in 0.95 s, 0.50 s and 0.51 s
→ `Cebu` (NONE), `COUNT` → 4.0, `SUM` → 3,623,462.0; `denotation_accuracy` 1.0 over those three) and a separate
64-row truncation probe (849 → 465 tokens by cell trimming, `rows_kept 64`, `truncated True`). Those runs exercised
the package, not this notebook, and are not notebook execution evidence.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `8ef61d5` / `08f1778d5ce6` | Kaggle CPU (`kurtvalcorza/dimer-nb2-tapas-table-qa` v1) | Default sample path | 258.7 s | **PASSED** — 8/8 ok code cells executed cleanly, 14 files, 1347 MB staged |

## Current status

The notebook source is complete and passes the static checks above; **no clean-runtime execution
has been recorded**, so the registry status is **Candidate** and the manual-evidence row is pending.
**The standalone carrier itself — executing the carried module cell in a runtime that has no repository
checkout — has been validated statically only (parity PASS) and never run end-to-end.** A carrier probe
did exec the install, carried-module and identity-assert cells in a fresh interpreter with the repository
package blocked on `sys.meta_path`, which confirms the cells define the public API without the package;
it fetched nothing and loaded no model. The clean run will therefore be the first execution of the
standalone path and of the staging path, and the first run of the pinned `pandas==3.0.5` install on a hosted
kernel (whose pre-imported pandas will trip the stale-import guard if it differs; use the fresh-container executor
in that case).
Promotion requires a reviewer to confirm a recorded run against the notebook blob under review and
an integrator to promote it; promotion is not performed by the builder. The commit that adds a
recorded-execution row changes documentation only; the executed source is the commit named in the
row.
