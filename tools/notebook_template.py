"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three
modules, carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the
generator from repository sources so they cannot drift from the package.

This template configures an E2E table-question-answering fine-tuning workflow: the pinned
google/tapas-large-finetuned-wtq snapshot is digest-verified and loaded, a digest-pinned WikiSQL shard is fetched
and its SQL programmes executed for gold cells and values, a table-disjoint sample is drawn and validated, the
synthetic city table is answered through the inference contract, the frozen model's denotation / operator / cell
accuracy is measured per question type beside two non-neural baselines, a bounded fine-tuning of the last encoder
blocks and the heads runs with the model's own weak-supervision loss, the held-out split is scored again, the city
table is re-answered, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "tapas_table_qa_pipeline",
    "repo_name": "tapas-table-question-answering-pipeline",
    "stem": "tapas_table_qa",
    "notebook_name": "tapas_table_qa_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "TAPASTableQAPipeline",
    "weights_key": "tapas-large-wtq",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers", "pandas"],
    "title": "TAPAS large (WTQ) — DIMER E2E table question answering fine-tuning tutorial (standalone)",
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
    "capability": "table question answering (one table of string cells + one question → selected cells, one aggregation operator NONE/SUM/AVERAGE/COUNT, and a pipeline-computed numeric answer) and bounded supervised fine-tuning of the last encoder blocks and heads on a labelled table-question set, using the pinned TAPAS-large WTQ weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `google/tapas-large-finetuned-wtq` snapshot (a 1.35 GB `model.safetensors`; no pickle is opened anywhere), fetches "
        "the pinned WikiSQL validation shard from the Hugging Face Hub (3.6 MB, refused on any byte-size or SHA-256 mismatch), "
        "executes its SQL programmes in pure Python for the gold cells and values, draws a table-disjoint sample of 240 / 90 / 150 "
        "questions stratified by operator, answers the synthetic city table through the inference contract with an input "
        "manifest and a rejection probe, measures the frozen model's denotation, operator and cell accuracy per question type "
        "over the 150 test questions beside the first-cell and keyword-lookup baselines, runs a bounded fine-tuning of the last "
        "two encoder blocks and the three heads with the model's own weak-supervision loss and validation-score epoch selection, "
        "scores the held-out questions again per type, re-answers the city table with the adapted model, exports the adapter "
        "as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify answer parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). TAPAS-large is a 24-layer encoder over up to 512 tokens: on the build workstation's CPU the "
        "whole path took about CPU_TOTAL_MIN minutes after the downloads (expect a multiple of that on a 2-vCPU hosted runtime); "
        "a CUDA runtime is used automatically when present and finishes in a few minutes."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one "
        "JSONL file of `{id, table, question, answer, category?}` records (Section 4 and the Prerequisites state the shape) — "
        "at least eight questions over at least two tables. They pass through the same validation, table-disjoint split, "
        "baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the WikiSQL sample. Uploaded "
        "tables stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "`google/tapas-large-finetuned-wtq` is the TAPAS model of Herzig et al. (2020) — a 24-layer BERT-style encoder over a "
        "flattened table (`[CLS] question [SEP] header row + data rows [SEP]`, lower-cased WordPiece with row, column and "
        "numeric-rank ids per token) with a cell-selection head and an aggregation head — fine-tuned by Google on "
        "WikiTableQuestions after SQA and WikiSQL; 336,734,214 parameters, published under the **Apache-2.0** licence. A cell "
        "is **selected** when the mean sigmoid probability over its tokens exceeds `CELL_THRESHOLD` (0.5), the operator is an "
        "`argmax` over `NONE`, `SUM`, `AVERAGE`, `COUNT`, and **the numeric answer is the pipeline's arithmetic over the selected "
        "cell strings**, never a model output. Neither decision is a calibrated probability, the model never abstains, and it "
        "applies no acceptance threshold — **every question yields some cells and some operator**.\n\n"
        "What this notebook adds to inference is **adaptation with labelled questions**. The dataset is real and shifted from "
        "the checkpoint's own training data: **WikiSQL** (Zhong et al. 2017, **BSD-3-Clause**) pairs Wikipedia tables with "
        "questions and the SQL programme that answers them; the pipeline executes each programme in pure Python to obtain the "
        "gold cells and value, maps WikiSQL's `MAX`/`MIN` to `NONE` with the extreme cell (TAPAS-WTQ has no such operators) "
        "and its `AVG` to `AVERAGE`, and keeps the original operator as the question's `category`. The WTQ checkpoint is "
        "already a strong lookup model on these tables (the build record measured denotation accuracy 0.82 frozen on the 150 "
        "test questions, 0.92 on lookups) but chooses the right operator for only about half the questions (0.57), so the "
        "honest question is narrow: does a bounded fine-tuning of the last two encoder blocks and the three heads on 240 "
        "questions — 80 lookups and 32 per operator, the model's own **weak-supervision loss** (the gold cells label a lookup, "
        "an operator question carries only its number) — move the **denotation accuracy**, the **aggregation accuracy** and "
        "the **cell accuracy** on a table-disjoint test split, per question type, against two **non-neural baselines** (the "
        "**first-cell floor** and a **keyword lookup**)? Nothing here is a quality claim about your tables: it is one seeded "
        "split of one sample.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (a 6-file manifest) — no pickle is opened anywhere "
        "in this notebook. Section 3 stages and digest-verifies those files before the tokenizer or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled table-question set, execute its programmes for gold answers, validate "
        "it and split it by table without leakage; answer a synthetic table through the public API and read the "
        "cell/operator/numeric contract correctly (fixed threshold, argmax, pipeline arithmetic, no abstention); measure the "
        "frozen model's denotation, operator and cell accuracy per question type beside two non-neural baselines; run a "
        "bounded fine-tuning with the model's own weak-supervision loss, explicit hyperparameters and validation-based epoch "
        "selection; evaluate on a table-disjoint test split; re-answer the synthetic table with the adapted model; and "
        "export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "conversational or multi-turn table QA (the SQA setting), tables with more than `MAX_ROWS` rows or `MAX_COLUMNS` "
        "columns, non-English tables, text-to-SQL or arbitrary computation beyond the four aggregation operators, the "
        "`MAX`/`MIN` operators themselves (WikiSQL's become a lookup of the extreme cell), fine-tuning of the embeddings or of "
        "any encoder block but the last ones, evaluation on WikiTableQuestions or WikiSQL proper (only one seeded "
        "480-question sample is scored here), and any claim that six question types on Wikipedia tables stand in for your "
        "tables. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (also float32). CPU is slow: TAPAS-large answers a question in about 0.85 s on the build workstation's CPU (the build record measured CPU_FROZEN_EVAL s to score the 150 test questions and CPU_ADAPT s for the four epochs of fine-tuning over 240 questions with per-epoch validation scoring); the whole default path took CPU_TOTAL s there with the snapshot and the shard already cached, and GPU_TOTAL s on an RTX 5070 Ti. A 2-vCPU hosted runtime will take a multiple of the workstation figure. The pinned `torch==2.14.0` install and the 1.35 GB checkpoint are the large downloads of the run; the WikiSQL shard adds 3.6 MB.",
        "- **Knowledge:** basic Python; what a sigmoid threshold and an argmax are and why neither is a calibrated probability; that an aggregation over selected cells is arithmetic the pipeline performs, not a model output; what denotation accuracy measures and why one seeded split gives no dispersion.",
        "- **Data contract:** records are `{id, table, question, answer}` with an optional `category` — `table` is `{column: [cells]}` with every header and cell a str (at most `MAX_ROWS` = 64 rows, `MAX_COLUMNS` = 32 columns, `MAX_CELL_CHARS` = 200 characters per cell), `question` a non-empty str of at most `MAX_QUERY_CHARS` = 500 characters, `answer` = `{aggregation: NONE|SUM|AVERAGE|COUNT, coordinates: [[row, col], ...], denotation: [cell, ...] | number}` where a `NONE` denotation is the selected cells and an operator denotation is the number the operator produces from them (validated). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..20,000 records; splitting is by table (normalised content) so no table lands in two splits; a table whose gold cells fall past the `MAX_TOKENS` = 512 truncation point is refused at training time. BYOD accepts one JSONL file (one record per line) or a JSON list in that shape.",
        "- **Validation is structural, not semantic:** every table, question and answer is checked for shape and arithmetic consistency, but nothing checks that an answer is right — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path fetches one parquet file from the Hugging Face Hub dataset repository `Salesforce/wikisql` at the immutable revision `48cfb60afd0d5f9d2231ca90f76edf9f975181bc` (`default/validation/0000.parquet`, 3,630,670 bytes, SHA-256 `ed524cc7…`, pinned in the carried `samples.py` and refused on any mismatch). WikiSQL is BSD-3-Clause (Zhong, Xiong & Socher 2017); nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. WikiSQL sample and split\n\n"
                "`fetch_corpus` returns the pinned shard from the cache under `weights/wikisql/` or the Hub — every cached or "
                "fetched copy is refused on a byte-count or SHA-256 mismatch — and `read_corpus` decodes its `question`, "
                "`table` and `sql` columns (8,421 questions over 2,630 tables). `wikisql_candidates` keeps the questions whose "
                "table fits the sample ceilings (at most 20 rows, 12 columns and 1,000 characters, so nothing is truncated at "
                "512 tokens) and whose SQL programme executes to a usable answer, mapping `MAX`/`MIN` to a `NONE` lookup of "
                "the extreme cell and `AVG` to `AVERAGE`; `build_sample_dataset` shuffles the tables, assigns each to one "
                "split first, then draws 80 lookups and 32 of each operator for training and 15 / 25 per operator for "
                "validation / test (240 / 90 / 150). `validate_dataset` checks every record against the contract — including "
                "that each operator denotation is what the pipeline's arithmetic produces from the gold cells — "
                "`check_split_disjoint` asserts no table (by normalised content) is shared, and the training split is written "
                "to `outputs/{stem}_train.jsonl` in the BYOD shape.\n\n"
                "Look for: 4,972 candidates, the six categories with 80 / 32 in training and 15 / 25 elsewhere, three digests, "
                "and four refusal probes — a duplicate id, a coordinate outside the table, an operator denotation that "
                "disagrees with its cells, and a dataset too small to use — each rejected before the model does anything."
            ),
            "code": (
                "import csv\n"
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / 'byod.jsonl'\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    shard = fetch_corpus(cache_dir='weights/wikisql')\n"
                "    corpus = read_corpus(shard)\n"
                "    candidates = wikisql_candidates(corpus)\n"
                "    splits = build_sample_dataset(candidates, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_REPO}} @ {{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'shard_bytes': len(shard), 'questions': len(corpus), 'tables': len({{r['table']['id'] for r in corpus}}), 'candidates': len(candidates)}}\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_jsonl(train_records, 'outputs/{stem}_train.jsonl')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'tables': manifest['n_tables'], 'categories': manifest['category_counts'], 'aggregations': manifest['aggregation_counts'], 'table_shape': manifest['table_shape'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{'id': example['id'], 'category': example.get('category'), 'question': example['question'], 'sql': example.get('sql'), 'answer': example['answer'], 'columns': list(example['table']), 'rows': len(next(iter(example['table'].values())))}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'coordinate outside the table': [{{**train_records[0], 'answer': {{**train_records[0]['answer'], 'coordinates': [[999, 0]]}}}}, *train_records[1:8]],\n"
                "    'operator denotation disagrees with its cells': [{{**r, 'answer': {{**r['answer'], 'denotation': 1e9}}}} if r['answer']['aggregation'] != 'NONE' else r for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Answer the synthetic table through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a 4-row × 3-column table of "
                "Philippine cities authored in code (the model card's smoke table, population cells with thousands "
                "separators) and three questions — a **lookup**, a **COUNT** and a **SUM** — with the answers the author "
                "expects; a different table family from WikiSQL, and questions the adapted model will answer again in "
                "Section 9. `validate_inputs` applies exactly the checks `answer` applies (`MAX_ROWS`, `MAX_COLUMNS`, "
                "`MAX_CELL_CHARS`, `MAX_QUERY_CHARS`; `MAX_TOKENS` is applied by the tokenizer and *truncates*, reported by "
                "`answer`) and returns an input manifest; a table carrying a numeric (non-string) cell is validated too and "
                "its rejection recorded as a finding. `answer(table, query)` returns `cells`, `coordinates`, `aggregation`, "
                "`numeric_answer` with `numeric_answer_source`, `unparsed_cells`, `aggregation_logits`, `n_tokens`, "
                "`tokens_before_truncation`, `rows_kept`, `truncated` and the decision rule; the per-grid `evaluation_report` "
                "on three authored questions is `sample-sanity` — plumbing evidence, not a measurement; whether the model is "
                "*right* is what Section 6 measures on 150 questions."
            ),
            "code": (
                "ceilings = {{'MAX_ROWS': MAX_ROWS, 'MAX_COLUMNS': MAX_COLUMNS, 'MAX_TOKENS': MAX_TOKENS, 'MAX_QUERY_CHARS': MAX_QUERY_CHARS, 'MAX_CELL_CHARS': MAX_CELL_CHARS, 'AGGREGATIONS': AGGREGATIONS, 'CELL_THRESHOLD': CELL_THRESHOLD, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'device': pipe.device}}\n"
                "print(ceilings)\n"
                "table = {{\n"
                "    'City': ['Manila', 'Cebu', 'Davao', 'Baguio'],\n"
                "    'Population (2020)': ['1,846,513', '964,169', '1,776,949', '366,358'],\n"
                "    'Region': ['NCR', 'Region VII', 'Region XI', 'CAR'],\n"
                "}}\n"
                "queries = ['Which city is in Region VII?', 'How many cities are listed?', 'What is the total population of Manila and Davao?']\n"
                "golds = ['Cebu', 4, 3623462]\n"
                "query_ids = [f'q{{index + 1}}' for index in range(len(queries))]\n"
                "sample_sha256 = hashlib.sha256(json.dumps({{'table': table, 'queries': queries}}, sort_keys=True).encode('utf-8')).hexdigest()\n"
                "input_manifest = validate_inputs(table, queries, names=query_ids)\n"
                "try:\n"
                "    validate_inputs({{'Item': ['a', 'b'], 'Units': [1, 2]}}, queries)\n"
                "except TypeError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'numeric-cell-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'sample_sha256': sample_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def answer_table(pipeline, label):\n"
                "    rows = []\n"
                "    n_rows = len(next(iter(table.values())))\n"
                "    for query_id, query in zip(query_ids, queries, strict=True):\n"
                "        started = time.perf_counter()\n"
                "        result = pipeline.answer(table, query)\n"
                "        elapsed = time.perf_counter() - started\n"
                "        checks = {{\n"
                "            'aggregation_known': result['aggregation'] in AGGREGATIONS,\n"
                "            'coordinates_inside_table': all(0 <= r < n_rows and 0 <= c < len(table) for r, c in result['coordinates']),\n"
                "            'cells_match_coordinates': result['cells'] == [list(table.values())[c][r] for r, c in result['coordinates']],\n"
                "            'count_equals_cells': result['aggregation'] != 'COUNT' or result['numeric_answer'] == len(result['cells']),\n"
                "            'n_tokens_within_ceiling': 1 <= result['n_tokens'] <= MAX_TOKENS,\n"
                "        }}\n"
                "        if not all(checks.values()):\n"
                "            raise RuntimeError(f'answer output failed a sanity check for {{query_id}}: {{checks}}')\n"
                "        rows.append({{'id': query_id, 'query': query, 'model': label, 'seconds': round(elapsed, 3), 'checks': checks, **result}})\n"
                "        print(f\"{{label}} {{query_id}}: {{result['aggregation']:<8}} cells={{result['cells']}} numeric_answer={{result['numeric_answer']}} tokens={{result['n_tokens']}} truncated={{result['truncated']}} seconds={{elapsed:.2f}}\")\n"
                "    return rows\n\n\n"
                "frozen_rows = answer_table(pipe, 'frozen')\n"
                "frozen_scene = evaluation_report(frozen_rows, golds, sample_kind='synthetic (authored in this notebook; the model card smoke table)')\n"
                "print({{'decision_rule': DECISION_RULE, 'numeric_answer_source': NUMERIC_ANSWER_SOURCE}})\n"
                "print({{'frozen_scene': {{m['id']: round(m['value'], 3) for m in frozen_scene['metrics']}}, 'verdict': frozen_scene['verdict']}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test questions\n\n"
                "Three systems frame the adaptation, each read three ways by `denotation_metrics` (carried in `metrics.py`): "
                "**denotation accuracy** (the WTQ criterion — a number to 1e-6 for an operator, otherwise the same multiset of "
                "cell strings), **aggregation accuracy** (the predicted operator equals the record's) and **cell accuracy** "
                "(the selected coordinates equal the record's as a set), overall and per question type. The **first-cell "
                "floor** answers every question with the first cell of the first column and `NONE`. The **keyword lookup** "
                "picks the column whose header shares the most words with the question, the rows whose other cells appear "
                "verbatim in it, and an operator from the question's wording (`how many` → `COUNT`, `total` → `SUM`, "
                "`average` → `AVERAGE`) — a table-QA system that never sees a token embedding. The **frozen model** is scored "
                "by `pipe.evaluate`, which answers every record through `answer` and scores the results. Expect the frozen "
                "model far above both baselines on denotation and lookups near the ceiling, and read the operator column: "
                "the build record measured 0.82 / 0.57 / 0.82 frozen, with `count` at 0.64 and lookups at 0.92."
            ),
            "code": (
                "METRICS = ('accuracy', 'aggregation_accuracy', 'cell_accuracy')\n"
                "baseline_first = first_cell_baseline(test_records)\n"
                "baseline_keyword = keyword_lookup_baseline(test_records)\n"
                "print({{'first_cell_baseline': {{k: round(baseline_first[k], 3) for k in METRICS}}, 'n': baseline_first['n'], 'note': baseline_first['baseline']}})\n"
                "print({{'keyword_lookup_baseline': {{k: round(baseline_keyword[k], 3) for k in METRICS}}, 'by_category': {{c: round(v['accuracy'], 2) for c, v in baseline_keyword['per_category'].items()}}, 'note': baseline_keyword['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'truncated': frozen_test['truncated'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'accuracy': round(v['accuracy'], 2), 'aggregation': round(v['aggregation_accuracy'], 2), 'cells': round(v['cell_accuracy'], 2)}} for c, v in frozen_test['per_category'].items()}}\n"
                "print({{'by_category_frozen': frozen_fields}})\n"
                "assert frozen_test['accuracy'] > baseline_keyword['accuracy'] > baseline_first['accuracy']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the last encoder blocks and the heads\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_LAYERS` encoder blocks and the three heads — the cell-selection "
                "weights, the column-selection weights and the aggregation classifier; two blocks by default, 25,198,598 of "
                "336,734,214 parameters — while the embeddings and every earlier block stay frozen. Supervision is the "
                "checkpoint's own **weak supervision**: a lookup labels its gold cells (the tokenizer turns the coordinates "
                "into token labels); an operator question carries only its number as `float_answer`, so the model's regression "
                "loss trains the operator and the soft cell selection together — labelling an operator question's cells too "
                "would let the model's aggregate mask re-route it to cell selection whenever it prefers `NONE`, which is the "
                "failure being adapted away. AdamW at a fixed learning rate, gradient clipping at 1.0, seeded shuffling, "
                "batches padded to their longest table, no scheduler. Epoch 0 records the frozen model's validation metrics; "
                "every epoch is scored on the 90 validation questions and the epoch with the highest validation **score** — "
                "the mean of denotation, aggregation and cell accuracy, a steadier selector than denotation accuracy alone on "
                "90 questions — is kept.\n\n"
                "Watch the training loss fall from about 3 to below 0.5 within four epochs while the validation aggregation "
                "accuracy jumps in the first epoch: the operator choice is what these questions teach. The build record's "
                "counter-examples — four blocks at the same rate, or a training draw with as many operator questions as "
                "lookups — traded lookup accuracy for operator accuracy; the default is the configuration that kept lookups "
                "whole."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in (*METRICS, 'score')}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_layers=TRAINABLE_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test questions' tables were never used for training or epoch selection, and no table appears in two "
                "splits. The adapted model is scored exactly as the frozen model was in Section 6, the four systems are put "
                "side by side on the three measures, and the per-type breakdown is repeated. Read it in this order: "
                "**aggregation accuracy** first (what the adaptation teaches — the build record measured 0.57 → 0.81), then "
                "**cell accuracy** (0.82 → 0.85) and **denotation accuracy** (0.82 → 0.84, three questions of 150), then the "
                "per-type rows, where `average`, `min` and `sum` each gained one question, `count` and `max` did not move and "
                "lookups stayed at 0.92. The cell asserts the adapted aggregation accuracy is above the frozen one; denotation "
                "accuracy is reported, not asserted, because on 150 questions it moves by single questions and the build "
                "record's other draws moved it either way. One seeded split gives **no dispersion estimate**; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a gain on six question types "
                "over Wikipedia tables says nothing about your tables until you measure them."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "adapted_fields = {{c: {{'n': v['n'], 'accuracy': round(v['accuracy'], 2), 'aggregation': round(v['aggregation_accuracy'], 2), 'cells': round(v['cell_accuracy'], 2)}} for c, v in adapted_test['per_category'].items()}}\n"
                "comparison = {{metric: {{'first_cell': round(baseline_first[metric], 3), 'keyword': round(baseline_keyword[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['by_category'] = {{c: {{'n': frozen_fields[c]['n'], 'frozen': frozen_fields[c]['accuracy'], 'adapted': adapted_fields[c]['accuracy'], 'frozen_aggregation': frozen_fields[c]['aggregation'], 'adapted_aggregation': adapted_fields[c]['aggregation']}} for c in frozen_fields}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'first_cell': {{k: v for k, v in baseline_first.items() if k != 'predictions'}}, 'keyword_lookup': {{k: v for k, v in baseline_keyword.items() if k != 'predictions'}}}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['aggregation_accuracy'] > frozen_test['aggregation_accuracy']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Re-answer the synthetic table, export the adapter and reload it\n\n"
                "The three city-table questions from Section 5 are answered again by the adapted model — a table family the "
                "adaptation never saw, so this is a small look at what the adaptation did *outside* its corpus (the build "
                "record's answers are in the model card; a changed answer here is a finding to record, not a failure) — "
                "reported with the per-grid `evaluation_report` (`sample-sanity`), and both answer sets are written to "
                "`outputs/{stem}_answers.csv`.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last two encoder blocks and the three heads, about "
                "100 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the training "
                "configuration and the epoch history (OUT8). `TAPASTableQAPipeline.from_artifact` re-verifies the base "
                "snapshot, checks the artifact manifest, its digest and its exact tensor set **before** deserialising, refuses "
                "any tensor outside the encoder blocks and heads, and overlays the tensors onto a freshly loaded base — a new "
                "object from files, not the in-memory model (VER2). The cell asserts identical cells, operators and "
                "aggregation logits on eight test questions (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "adapted_rows = answer_table(pipe, 'adapted')\n"
                "adapted_scene = evaluation_report(adapted_rows, golds, sample_kind='synthetic (authored in this notebook; the model card smoke table)')\n"
                "print({{'scene_after_adaptation': {{m['id']: round(m['value'], 3) for m in adapted_scene['metrics']}}, 'verdict': adapted_scene['verdict']}})\n"
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['model', 'id', 'query', 'aggregation', 'cells', 'coordinates', 'numeric_answer', 'numeric_answer_source', 'unparsed_cells', 'n_tokens', 'rows_kept', 'truncated', 'seconds'])\n"
                "    for r in [*frozen_rows, *adapted_rows]:\n"
                "        writer.writerow([r['model'], r['id'], r['query'], r['aggregation'], ' | '.join(r['cells']), json.dumps(r['coordinates']), r['numeric_answer'], r['numeric_answer_source'], ' | '.join(r['unparsed_cells']), r['n_tokens'], r['rows_kept'], r['truncated'], r['seconds']])\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = TAPASTableQAPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [pipe.answer(r['table'], r['question']) for r in test_records[:8]]\n"
                "after = [reloaded.answer(r['table'], r['question']) for r in test_records[:8]]\n"
                "parity = {{'identical_answers': sum(a['cells'] == b['cells'] and a['aggregation'] == b['aggregation'] and a['aggregation_logits'] == b['aggregation_logits'] for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_answers'] == parity['of']\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'file': CORPUS_FILE, 'bytes': CORPUS_BYTES, 'sha256': CORPUS_SHA256, 'license': CORPUS_LICENSE, 'sample_digest': SAMPLE_DIGEST}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'ceilings': ceilings, 'sample': {{'name': 'synthetic_cities_table', 'sample_sha256': sample_sha256, 'golds': golds}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene, 'answers_file': 'outputs/{stem}_answers.csv'}},\n"
                "    'decision_rule': DECISION_RULE,\n"
                "    'numeric_answer_source': NUMERIC_ANSWER_SOURCE,\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'pandas': pandas.__version__, 'device': pipe.device, 'dtype': 'float32'}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen WTQ checkpoint is already a strong lookup model on WikiSQL tables — far above the two non-neural "
        "baselines, at 0.92 on lookups in the build record — and a bounded fine-tuning of the last two encoder blocks and the "
        "heads on 240 questions moves what these questions teach: the operator choice (aggregation accuracy 0.57 → 0.81), a "
        "little of the cell selection (0.82 → 0.85) and three questions of denotation accuracy (0.82 → 0.84), with the lookups "
        "untouched and a 100 MB adapter that reloads to identical answers. That is the claim: the adaptation contract works "
        "end to end on a real labelled table-question set, and the numbers it produces are read on three measures, per "
        "question type, against two non-neural baselines and the frozen model rather than in isolation.\n\n"
        "The test split is 150 questions from one seeded draw of one sample, the validation split that picks the epoch is 90, "
        "denotation accuracy moves in steps of one question, and the build record's own draws show the estimate's fragility: "
        "four trainable blocks, or a training draw with as many operator questions as lookups, raised operator accuracy just "
        "as much while *costing* lookups, and on another draw denotation accuracy did not move at all. So a gain here says the "
        "contract works, not that the adapted model is better on your tables, that a sigmoid threshold or an argmax is a "
        "probability, or that the pipeline's arithmetic is right when the cells are wrong — it still returns some cells and "
        "some operator for every question, and it is confidently wrong when it is wrong. Fine-tuning on a narrow set can also "
        "erode the model elsewhere; the city table re-answered in Section 9 is three questions of evidence about that, not a "
        "measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the first-cell floor, the keyword lookup and the frozen "
        "model's score on *your* questions are the numbers to read before any adapted one, per question type. **Leakage:** "
        "keep every table in one split (the contract splits by normalised table content) and split by source when your "
        "tables come from few documents. **Supervision:** the gold cells of a lookup and the number of an operator question "
        "are the two shapes the weak-supervision loss understands; a table whose gold cells lie past the 512-token truncation "
        "point is refused at training time, and a mislabelled answer is learned without complaint.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled table-question set, "
        "validate the demonstrated dataset contract without leakage, execute the inference contract and a bounded fine-tuning, "
        "evaluate against two trivial baselines and the frozen model on a table-disjoint split, and emit the shown "
        "machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark superiority, "
        "denotation accuracy on any other domain or table shape, a usable threshold, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_LAYERS = 4` and compare the artifact "
        "size, the operator accuracy and the lookup accuracy; raise `EPOCHS` and watch the validation score pick the epoch "
        "while the training loss keeps falling; change `LEARNING_RATE` to `2e-5` and read a smaller, steadier gain; or bring "
        "your own JSONL through BYOD and read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/tapas-table-question-answering-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/tapas-table-question-answering-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/tapas-table-question-answering-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/tapas\n"
        "- TAPAS: Weakly Supervised Table Parsing via Pre-training (Herzig et al., 2020): https://arxiv.org/abs/2004.02349\n"
        "- Understanding tables with intermediate pre-training (Eisenschlos et al., 2020): https://arxiv.org/abs/2010.00571\n"
        "- Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning (WikiSQL; Zhong, Xiong & Socher, 2017): https://arxiv.org/abs/1709.00103 — dataset https://github.com/salesforce/WikiSQL (BSD-3-Clause)\n"
        "- Compositional Semantic Parsing on Semi-Structured Tables (WikiTableQuestions): https://arxiv.org/abs/1508.00305\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
