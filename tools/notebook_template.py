"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "tapas_table_qa_pipeline",
    "repo_name": "tapas-table-question-answering-pipeline",
    "stem": "tapas_table_qa",
    "notebook_name": "tapas_table_qa_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "TAPASTableQAPipeline",
    "weights_key": "tapas-large-wtq",
    "runtime_imports": ["torch", "transformers", "pandas"],
    "title": "TAPAS large (WTQ) — DIMER table question answering tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/tapas-table-question-answering-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/tapas-table-question-answering-pipeline/blob/main/tutorials/tapas_table_qa_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Ftapas--large--finetuned--wtq-ffcc4d?style=flat",
            "https://huggingface.co/google/tapas-large-finetuned-wtq",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-google--research%2Ftapas-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/google-research/tapas",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2004.02349-b31b1b.svg", "https://arxiv.org/abs/2004.02349"),
    ],
    "capability": "table question answering (one table of string cells + one question → selected cells, one aggregation operator NONE/SUM/AVERAGE/COUNT, and a pipeline-computed numeric answer) using the pinned TAPAS-large WTQ weights",
    "intro": (
        "At inference the tokenizer flattens the table to `[CLS] question [SEP] header row + data rows [SEP]` "
        "(lower-cased WordPiece, with row, column and numeric-rank ids per token) and one forward pass of the "
        "24-layer BERT-style encoder runs; a cell-selection head scores every token and a cell is **selected** when "
        "the mean sigmoid probability over its tokens exceeds `CELL_THRESHOLD` (0.5, the upstream default), while an "
        "aggregation head picks **one** of `NONE`, `SUM`, `AVERAGE`, `COUNT` by `argmax`. Those two decisions are "
        "exactly what the upstream `TapasTokenizer.convert_logits_to_predictions` produces and are all the model "
        "emits. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting "
        "— the pinned checkpoint is used as published. What the upstream checkpoint supplies is the encoder, the two "
        "heads and the tokenizer; what the carried pipeline module adds is manifest verification, table and question "
        "validation with named ceilings (`MAX_ROWS`, `MAX_COLUMNS`, `MAX_TOKENS`, …), the `answer` method with a fixed "
        "output contract, the `validate_inputs`/`evaluation_report` stage helpers, a `denotation_accuracy` metric, and "
        "**the numeric answer itself: `numeric_answer` is computed by the pipeline from the selected cell strings** "
        "(COUNT = number of cells; SUM/AVERAGE parse each cell back to a number, else `None`) and is labelled as "
        "such in `numeric_answer_source`. The model never emits a number."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author a synthetic table of "
        "string cells with three questions (or upload your own CSV and questions), stage and digest-verify the "
        "immutable upstream snapshot, surface the pipeline's ceilings and validate the table and questions into an "
        "input manifest, run table question answering and read the cell/aggregation/numeric contract correctly, "
        "read the machine-readable evaluation report (denotation accuracy on the author's gold answers as a sanity "
        "check, `not-measurable` without them), and export the answers as CSV plus a provenance JSON."
    ),
    "exclusions": (
        "conversational or multi-turn table QA (the SQA setting), tables with more than `MAX_ROWS` rows or "
        "`MAX_COLUMNS` columns, non-English tables, text-to-SQL or arbitrary computation beyond the four aggregation "
        "operators, fine-tuning, batch questions in one forward pass, or raw-logit access. The repository exposes none "
        "of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (also float32; the pipeline loads the checkpoint in float32 on both). The model card's CPU smoke loaded and verified the 1.35 GB snapshot in 6.83 s and answered three questions on a 4×3 table in 0.95 s, 0.50 s and 0.51 s, so the default runs in well under a minute on a hosted CPU runtime once the download finishes. The pinned `torch==2.14.0` install and the 1.35 GB `model.safetensors` are the largest downloads of the run.",
        "- **Knowledge:** basic Python; what a sigmoid threshold and an argmax are and why neither is a calibrated probability; that an aggregation over selected cells is arithmetic the pipeline performs, not a model output.",
        "- **Data:** the default sample is one synthetic 4-row × 3-column table and three questions authored in code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 CSV whose first row is the header and whose every cell is read as text, plus your questions typed into the form field. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded tables remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**, written in this cell: a 4-row × 3-column table of Philippine "
                "cities (the model card's smoke table) whose population cells carry thousands separators, and three "
                "questions each given a stable identifier — a cell **lookup**, a **COUNT** and a **SUM** — together "
                "with the answer the author expects. Every header and cell is a **string**: TAPAS tokenises text, so "
                "numbers must arrive as text and the pipeline parses them back only when it computes SUM/AVERAGE. "
                "The expected answers are the author's intent, not a labelled dataset: whether the model reproduces "
                "them is a sanity check that the code path works, never benchmark evidence.\n\n"
                "BYOD is optional and disabled by default. Expected BYOD input: one UTF-8 CSV whose first row is the "
                "header, at most `MAX_ROWS` data rows and `MAX_COLUMNS` columns, every cell read as text and at most "
                "`MAX_CELL_CHARS` characters; questions go in `BYOD_QUESTIONS`, separated by ` | `, each at most "
                "`MAX_QUERY_CHARS` characters. A flattened table longer than `MAX_TOKENS` WordPiece tokens is "
                "**truncated** by the tokenizer (cell text trimmed to a common token count, then trailing rows "
                "dropped) and the result reports it. The upload stays inside this runtime. If you also hold gold "
                "answers, keep them outside the notebook — Section 7 explains what to compute with them."
            ),
            "code": (
                "import csv\n"
                "import hashlib\n"
                "import io\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_QUESTIONS = 'How many rows are there? | What is the total of the second column?'  # @param {{type:\"string\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    reader = list(csv.reader(io.StringIO(uploaded[sample_name].decode('utf-8'))))\n"
                "    if len(reader) < 2:\n"
                "        raise ValueError(f'{{sample_name}}: expected a header row followed by at least one data row')\n"
                "    header, data_rows = reader[0], reader[1:]\n"
                "    table = {{name: [row[i] if i < len(row) else '' for row in data_rows] for i, name in enumerate(header)}}\n"
                "    queries = [q.strip() for q in BYOD_QUESTIONS.split('|') if q.strip()]\n"
                "    golds = None\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    table = {{\n"
                "        'City': ['Manila', 'Cebu', 'Davao', 'Baguio'],\n"
                "        'Population (2020)': ['1,846,513', '964,169', '1,776,949', '366,358'],\n"
                "        'Region': ['NCR', 'Region VII', 'Region XI', 'CAR'],\n"
                "    }}\n"
                "    queries = [\n"
                "        'Which city is in Region VII?',\n"
                "        'How many cities are listed?',\n"
                "        'What is the total population of Manila and Davao?',\n"
                "    ]\n"
                "    golds = ['Cebu', 4, 3623462]\n"
                "    sample_name = 'synthetic_cities_table'\n"
                "    sample_kind = 'synthetic (authored in this cell; the model card smoke table)'\n"
                "query_ids = [f'q{{index + 1}}' for index in range(len(queries))]\n"
                "sample_sha256 = hashlib.sha256(json.dumps({{'table': table, 'queries': queries}}, sort_keys=True).encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'columns': list(table), 'rows': len(next(iter(table.values()))), 'questions': len(queries), 'golds': golds, 'sample_sha256': sample_sha256}})\n"
                "for query_id, query in zip(query_ids, queries, strict=True):\n"
                "    print(f'{{query_id}}: {{query[:120]}}')"
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it takes the table and the list of "
                "questions `answer` will be called with, and its checks are the method's own — `_check_table` and "
                "`_check_query` — so a rejection here is a rejection there. `MAX_ROWS` (64) and `MAX_COLUMNS` (32) are "
                "the pinned config's `max_num_rows`/`max_num_columns` and **reject** larger tables; `MAX_CELL_CHARS` "
                "and `MAX_QUERY_CHARS` are character guards applied before tokenisation; `MAX_TOKENS` (512, the "
                "checkpoint's fine-tuning sequence length) is applied by the tokenizer after tokenisation and "
                "**truncates** rather than rejects, so it is reported by `answer` (`truncated`, `rows_kept`, "
                "`tokens_before_truncation`) and cannot be observed at this stage; `AGGREGATIONS` names the four "
                "operators and `CELL_THRESHOLD` the selection cut-off. The manifest records the table shape and how "
                "many cells parse as numbers, and is written to `outputs/{stem}_input_manifest.json`. To show what "
                "rejection looks like, the cell also validates a table carrying a numeric (non-string) cell and "
                "records the pipeline's own error message as a finding. The notebook never trims or alters the table."
            ),
            "code": (
                "os.makedirs('outputs', exist_ok=True)\n"
                "ceilings = {{'MAX_ROWS': MAX_ROWS, 'MAX_COLUMNS': MAX_COLUMNS, 'MAX_TOKENS': MAX_TOKENS, 'MAX_QUERY_CHARS': MAX_QUERY_CHARS, 'MAX_CELL_CHARS': MAX_CELL_CHARS, 'AGGREGATIONS': AGGREGATIONS, 'CELL_THRESHOLD': CELL_THRESHOLD}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs(table, queries, names=query_ids)\n"
                "# Demonstrate the strings-only rejection; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs({{'Item': ['a', 'b'], 'Units': [1, 2]}}, queries)\n"
                "except TypeError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'numeric-cell-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "print({{'token_ceiling': f'MAX_TOKENS={{MAX_TOKENS}} is applied by the tokenizer after tokenisation and truncates (cell text trimmed, then trailing rows dropped); answer() reports truncated, rows_kept and tokens_before_truncation'}})"
            ),
        },
        {
            "md": (
                "## 6. Answer the questions and read the contract correctly\n\n"
                "**Input/output contract.** `answer(table, query)` takes the table and one question and returns "
                "`cells` (the selected cell strings, in row-major order), `coordinates` (`[row, column]` into the data "
                "rows, 0 = first row below the header), `aggregation` (one of `AGGREGATIONS`), `answer` (the upstream "
                "pipeline's string form: the cells joined with `, `, prefixed by `SUM > ` etc. when an operator was "
                "chosen), `numeric_answer` with `numeric_answer_source`, `unparsed_cells` (cells that could not be "
                "read as numbers, in which case `numeric_answer` is `None`), `aggregation_logits` (the four raw head "
                "scores), `n_tokens`, `tokens_before_truncation`, `rows_kept`, `truncated`, the decision rule, the "
                "device and the model identity. **Decision semantics:** the cell decision is a fixed sigmoid "
                "threshold (`CELL_THRESHOLD` = 0.5) and the operator decision is an `argmax` over four logits; "
                "neither is a calibrated probability, the pipeline applies no acceptance threshold on the aggregation "
                "logits, and a question the table cannot answer still yields *some* cells and *some* operator — "
                "including `SUM` over a single cell, which the model card's probe observed. **The numeric answer is "
                "the pipeline's arithmetic**, not a model output: it is exactly right when the selected cells and "
                "operator are right, and confidently wrong otherwise. The model card's CPU smoke on this table "
                "returned `Cebu` (NONE), `COUNT` over the four city cells → 4.0, and `SUM` over `1,846,513` and "
                "`1,776,949` → 3,623,462.0; those are single observations, not expected values. Inference is "
                "deterministic on a fixed device and dtype (`model.eval()`, no sampling, no seed needed)."
            ),
            "code": (
                "import time\n\n"
                "results = []\n"
                "n_rows = len(next(iter(table.values())))\n"
                "for query_id, query in zip(query_ids, queries, strict=True):\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.answer(table, query)\n"
                "    elapsed = time.perf_counter() - started\n"
                "    checks = {{\n"
                "        'aggregation_known': result['aggregation'] in AGGREGATIONS,\n"
                "        'coordinates_inside_table': all(0 <= r < n_rows and 0 <= c < len(table) for r, c in result['coordinates']),\n"
                "        'cells_match_coordinates': result['cells'] == [list(table.values())[c][r] for r, c in result['coordinates']],\n"
                "        'count_equals_cells': result['aggregation'] != 'COUNT' or result['numeric_answer'] == len(result['cells']),\n"
                "        'numeric_none_iff_unparsed_or_no_aggregation': (result['numeric_answer'] is None) == (bool(result['unparsed_cells']) or result['aggregation'] == 'NONE' or not result['cells']),\n"
                "        'rows_kept_within_table': 0 <= result['rows_kept'] <= n_rows,\n"
                "        'n_tokens_within_ceiling': 1 <= result['n_tokens'] <= MAX_TOKENS,\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'answer output failed a sanity check for {{query_id}}: {{checks}}')\n"
                "    results.append({{'id': query_id, 'query': query, 'seconds': round(elapsed, 3), 'checks': checks, **result}})\n"
                "    print(f\"{{query_id}}: {{query}}\")\n"
                "    print(f\"    {{result['aggregation']:<8}} cells={{result['cells']}} coords={{result['coordinates']}} numeric_answer={{result['numeric_answer']}} ({{result['numeric_answer_source']}})\")\n"
                "    print(f\"    tokens={{result['n_tokens']}}/{{result['tokens_before_truncation']}} rows_kept={{result['rows_kept']}}/{{n_rows}} truncated={{result['truncated']}} seconds={{elapsed:.3f}}\")\n"
                "print({{'decision_rule': DECISION_RULE, 'numeric_answer_source': NUMERIC_ANSWER_SOURCE}})\n"
                "if any(r['truncated'] for r in results):\n"
                "    print('At least one question ran on a truncated table: cell text was trimmed and/or trailing rows dropped to fit MAX_TOKENS; answers may refer to a table the model only partly saw.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. The "
                "repository ships one metric helper, **`denotation_accuracy`** — the WikiTableQuestions criterion: a "
                "prediction is correct when its denotation equals the gold (a numeric gold against `numeric_answer`, "
                "or the single selected cell parsed as a number; otherwise the selected cell strings against the gold "
                "strings after lower-casing and whitespace collapse). On the synthetic sample the author's three "
                "expected answers are passed as `golds`, so the verdict is **`sample-sanity`** with a single-sample "
                "estimate and no dispersion — a falsifiable plumbing check on three questions, **not** an accuracy "
                "claim and not comparable to the upstream WTQ dev figure (0.5097, upstream-reported, not measured "
                "here). On a BYOD upload no golds are supplied, so the verdict is `not-measurable` and the report "
                "states what would make it measurable: one gold denotation per question over enough questions from "
                "your own tables to state a dispersion. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(results, golds, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'sample-sanity':\n"
                "    print('denotation_accuracy on the author-supplied golds is a plumbing check on a handful of questions, not an accuracy figure; see needs.')\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is reported: supply one gold denotation per question to compute denotation_accuracy on your own table-question set.')"
            ),
        },
        {
            "md": (
                "## 8. Export answers and provenance\n\n"
                "Two further files are written under `outputs/` beside the input manifest and the evaluation report. "
                "The answers go to CSV (`outputs/{stem}_answers.csv`) with one row per question — `id`, `query`, "
                "`aggregation`, `cells` (joined with ` | `), `coordinates`, `numeric_answer`, `numeric_answer_source`, "
                "`unparsed_cells`, `n_tokens`, `tokens_before_truncation`, `rows_kept`, `truncated`, `seconds` — so "
                "every answer stays attached to its question. One JSON record (`outputs/{stem}_result.json`) "
                "preserves the table, every result with its aggregation logits and sanity checks, the decision rule, "
                "the ceilings in force, the input manifest, the evaluation report, the sample identity and digest, "
                "the notebook's source (repository, revision, embedded module digest, generator), the model "
                "identifier, the immutable model revision, the model licence, the verified snapshot summary, and the "
                "runtime identity (Python, `torch`, `transformers`, `pandas`, device, dtype). No credentials are "
                "involved in any step, so none can reach the export."
            ),
            "code": (
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['id', 'query', 'aggregation', 'cells', 'coordinates', 'numeric_answer', 'numeric_answer_source', 'unparsed_cells', 'n_tokens', 'tokens_before_truncation', 'rows_kept', 'truncated', 'seconds'])\n"
                "    for r in results:\n"
                "        writer.writerow([r['id'], r['query'], r['aggregation'], ' | '.join(r['cells']), json.dumps(r['coordinates']), r['numeric_answer'], r['numeric_answer_source'], ' | '.join(r['unparsed_cells']), r['n_tokens'], r['tokens_before_truncation'], r['rows_kept'], r['truncated'], r['seconds']])\n"
                "payload = {{\n"
                "    'table': table,\n"
                "    'results': [{{key: value for key, value in r.items() if key not in ('device', 'source', 'model_id', 'model_revision')}} for r in results],\n"
                "    'decision_rule': DECISION_RULE,\n"
                "    'numeric_answer_source': NUMERIC_ANSWER_SOURCE,\n"
                "    'answers_file': 'outputs/{stem}_answers.csv',\n"
                "    'ceilings': ceilings,\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'sample_sha256': sample_sha256, 'golds': golds}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': snapshot['path'], 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes')}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'pandas': pandas.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "TAPAS answers a question by selecting cells (a fixed 0.5 sigmoid threshold per cell) and choosing one of four "
        "aggregation operators (an argmax); it emits no number and no calibrated confidence. The `numeric_answer` you "
        "see is the pipeline's arithmetic over the selected cell strings — exact when the selection and operator are "
        "right, silently wrong when they are not, and `None` when a selected cell does not parse as a number. The "
        "model always produces some answer, including operators applied to a single cell, and applies no acceptance "
        "threshold on the aggregation logits; any cut-off is the caller's to set on labelled questions from their own "
        "tables. Tables are lower-cased WordPiece text: numbers must arrive as strings, and a flattened table over 512 "
        "tokens is truncated (cell text trimmed, then trailing rows dropped) rather than rejected, so `truncated`, "
        "`rows_kept` and `tokens_before_truncation` must be read before trusting an answer. The checkpoint was "
        "fine-tuned on English Wikipedia tables (WTQ, after SQA and WikiSQL); behaviour on other domains, languages or "
        "table shapes is not measured here. On the synthetic sample the sample-sanity denotation accuracy is a "
        "plumbing check on three questions, not an accuracy figure; a real evaluation needs gold denotations over "
        "enough questions to state a dispersion.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model snapshot, validate the demonstrated inputs against the enforced "
        "ceilings, execute the public pipeline path, and emit the shown machine-readable outputs in the tested runtime "
        "— without the repository being reachable. It does **not** establish benchmark superiority, denotation "
        "accuracy on any domain, a usable threshold, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments.** Ask a question the table cannot answer and read which cells and operator the model "
        "still returns; ask for an average and watch the pipeline parse the thousands separators; paste a table with a "
        "long text column and read `truncated`/`rows_kept` to see what the tokenizer dropped; assemble a dozen "
        "questions with gold denotations over one of your own tables and compute the `denotation_accuracy` the "
        "evaluation report asks for. None of these turns the sample result into evidence of production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/tapas-table-question-answering-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/tapas-table-question-answering-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/tapas-table-question-answering-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/tapas\n"
        "- TAPAS: Weakly Supervised Table Parsing via Pre-training: https://arxiv.org/abs/2004.02349\n"
        "- Understanding tables with intermediate pre-training: https://arxiv.org/abs/2010.00571\n"
        "- Compositional Semantic Parsing on Semi-Structured Tables (WikiTableQuestions): https://arxiv.org/abs/1508.00305"
    ),
}
