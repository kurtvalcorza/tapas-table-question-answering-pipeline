"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.5 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three
modules, carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the
generator from repository sources so they cannot drift from the package.

This template configures an E2E table-question-answering fine-tuning workflow: the pinned
google/tapas-large-finetuned-wtq snapshot is digest-verified and loaded, digest-pinned TAT-QA and FinQA files are
fetched and converted to gold cells and values (lookups from TAT-QA, sums and averages from FinQA), a group-disjoint
sample is drawn and validated, the synthetic city table is answered through the inference contract, the frozen
model's value-equality / strict-operator / operator / cell accuracy is measured per question type beside two
non-neural baselines, a bounded fine-tuning of the last encoder
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
    "notebook_kind": "TUTORIAL",
    # NOTEBOOK_SPEC 2.5 §3.6.5 canonical use notice (DOC17-DOC21); this notebook has a BYOD path, so the data clause stays.
    "use_notice": (
        "> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,\n"
        "> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party\n"
        "> model weights and datasets under their own licenses (see the model card) and uses compute\n"
        "> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are\n"
        "> responsible for having the right to use it. Results here are tutorial evidence, not\n"
        "> benchmarks. Everything is provided \"as is\", without warranty of any kind. To the maximum\n"
        "> extent permitted by law, the maintainers and the DIMER platform accept no liability for\n"
        "> its use."
    ),
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
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
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned `google/tapas-large-finetuned-wtq` snapshot (a 1.35 GB `model.safetensors`; no pickle is opened anywhere), fetches "
        "four pinned table-QA files that are not in the checkpoint's fine-tuning chain — TAT-QA from the Hugging Face Hub and FinQA "
        "from GitHub, each at an immutable commit (about 116 MB together, each refused on any byte-size or SHA-256 mismatch) — converts "
        "their answers to gold cells and values, draws a sample of 200 / 60 / 105 questions in which no source report or table spans "
        "two splits (lookups, sums and averages; **COUNT is not exercised in this revision**), answers the synthetic city table through "
        "the inference contract with an input manifest and a rejection probe, measures the frozen model's value-equality and "
        "strict-operator accuracy (co-primary) plus operator and cell accuracy per question type over the 105 test questions beside "
        "the first-cell and keyword-lookup baselines, runs a bounded fine-tuning of the last "
        "two encoder blocks and the three heads with the model's own weak-supervision loss and validation-score epoch selection, "
        "scores the held-out questions again per type, re-answers the city table with the adapted model, exports the adapter "
        "as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify answer parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.5 §5). TAPAS-large is a 24-layer encoder over up to 512 tokens: a CUDA runtime is used automatically "
        "when present; on CPU expect well over half an hour. No hosted run of this revision is recorded yet; the previous revision's "
        "default path (a larger WikiSQL sample, blob `f8e0d361`) took 458.9 s wall in its Colab CLI T4 run of 2026-10-10 — quoted from "
        "that run, not re-run here."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a JSONL file in "
        "the runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one file in Colab, then choose **Run after** from "
        "that cell (it first puts the model back to the pinned base) to supply one JSONL file of `{id, table, question, answer, "
        "category?}` records (Section 4 and the Prerequisites state the shape) — at least **12 questions: one per table over 12 "
        "tables, or two per table over 6 tables** (the split is by table, and the training tables must hold 8 questions). They pass through the same validation, table-disjoint split, "
        "baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the default sample. Uploaded "
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
        "What this notebook adds to inference is **adaptation with labelled questions** on tables the checkpoint was not "
        "fine-tuned on: lookups from **TAT-QA** and sums and averages from **FinQA**, both over company financial reports "
        "(Data context below). The question is narrow: does a bounded fine-tuning of the last two encoder blocks and the three "
        "heads on 200 questions — 80 lookups, 60 sums and 60 averages, trained with the model's own **weak-supervision loss** (the "
        "gold cells label a lookup, an operator question carries only its number) — move **value-equality accuracy** and "
        "**strict-operator accuracy** (the two co-primary answer rules), and the **aggregation accuracy** and **cell accuracy** "
        "diagnostics, on a split in which no source report or table is shared, per question type, against two **non-neural "
        "baselines** (the **first-cell floor** and a **keyword lookup**)? **COUNT is not exercised in this revision:** neither "
        "source has questions that convert cleanly to `COUNT`, so no result here says anything about that operator. Nothing "
        "here is a quality claim about your tables: it is one seeded split of one sample.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (a 6-file manifest) — no pickle is opened anywhere "
        "in this notebook. Section 3 stages and digest-verifies those files before the tokenizer or the model is constructed."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter, and wants to see how a pretrained table-QA model is fine-tuned with a small labelled set — what it learns, what it can unlearn, and how to tell from held-out numbers per question type. No prior experience with TAPAS or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. A GPU runtime is strongly recommended; on CPU the run takes over half an hour.\n\n**Input → Model → Output.**\n\n| | Answering | Fine-tuning |\n|---|---|---|\n| Input | one table of string cells (≤ 64 rows, ≤ 32 columns) and one question | labelled questions: 200 training and 60 validation questions (TAT-QA lookups, FinQA sums and averages), split so no report or table is shared |\n| Model | TAPAS-large (WTQ): a 24-layer encoder over the flattened table with a cell-selection head and an aggregation head | the last two encoder blocks and the three heads trained with the model's own weak-supervision loss |\n| Output | selected cells, one operator (`NONE`, `SUM`, `AVERAGE`, `COUNT`) and the pipeline's arithmetic over the cells — never an abstention | a safetensors adapter, and value-equality, strict-operator, operator and cell accuracy per question type beside two baselines |\n\n**How to use this notebook.** Choose a runtime (a GPU is strongly recommended), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer. No hosted run of this revision is recorded yet, so the worked answers say what to look for and quote no number from this revision; a number from an earlier revision is labelled with that revision and was not re-run here. Section 10 is a **change-one-thing experiment**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the financial-table sample and a report-level split *(evaluation practice)* → 5 the inference contract on a synthetic table *(core concept: cells, operator, arithmetic)* → 6 two baselines and the frozen model per question type *(evaluation practice)* → 7 bounded fine-tuning *(core concept: weak supervision)* → 8 held-out evaluation per type → 9 re-answer, export and reload *(engineering)* → 10 change one thing (optional) → conclude."
            ),
            (
                "## Problem statement\n\n"
                "Analysts, auditors and students who read company financial reports ask the same kinds of question of their tables: "
                "*what was X in 2019*, *what is the total of X over these years*, *what is the average of X*. A table-QA model answers "
                "by selecting cells and choosing an operator, so a deployment that adopts one needs to know two things on its own "
                "tables before trusting it: does the model pick the right **cells**, and does it pick the right **operator**? This "
                "notebook shows how to measure both on a labelled sample, how a small bounded fine-tuning changes them, and how to "
                "read the change per question type rather than as one number. It teaches the method; it does not certify a model "
                "for financial use."
            ),
            (
                "## Data context\n\n"
                "The default sample comes from two public table-QA datasets built from **company financial reports** (annual "
                "reports and SEC filings), not from Wikipedia:\n\n"
                "- **TAT-QA** [5] — the `lookup` questions. Hybrid table-and-text questions written by finance-literate annotators; "
                "only questions answered from the table by one or more cell spans are kept. Licence **CC BY 4.0**; Hub dataset "
                "`next-tat/TAT-QA` at revision `c96247f5077e`, file `tatqa_dataset_train.json`.\n"
                "- **FinQA** [6] — the `sum` and `average` questions. Questions with an executable reasoning programme over a "
                "report's table and text; only table-only programmes that are a chain of additions (`SUM`), that chain divided "
                "by its number of terms (`AVERAGE`), or `table_sum` / `table_average` over one row are kept, and every term must "
                "match exactly one table cell. Licence **MIT** on the official GitHub repository (`czyssrs/FinQA` at commit "
                "`0f16e2867bef`; train, dev and test files pooled); the Hugging Face card of `ibm-research/finqa` says **CC BY 4.0**, "
                "so this notebook follows the stricter reading and attributes it as CC BY 4.0 too.\n\n"
                "**Conversion and filtering.** The first table row becomes the header (blank names filled, duplicates suffixed); "
                "cells are whitespace-collapsed and currency or negative formats are rewritten so the pipeline's number parser "
                "reads them (`$ 1,234` → `1,234`, `( 12 )` → `-12`). A table must fit the sample ceilings — at most 20 rows, 12 "
                "columns and 1,000 characters — so nothing is truncated at 512 tokens. On the pinned files this keeps **1,244 of "
                "2,055** TAT-QA table-span questions (1,965 pass the table ceilings; the rest lose their answer cell, which is a "
                "header or occurs more than once) and **366 of 501** FinQA table-only sum/average programmes (205 sums, 161 "
                "averages; 496 pass the table ceilings, the rest fail the one-cell-per-term match). Section 4 prints this funnel.\n\n"
                "**Overlap with the checkpoint (DAT9).** The pinned checkpoint was fine-tuned in a chain on SQA, WikiSQL and "
                "WikiTableQuestions, all over Wikipedia tables; neither TAT-QA nor FinQA draws its tables from Wikipedia, and both "
                "were released (2021) after the checkpoint, so **neither source overlaps SQA, WikiSQL or WTQ**. The BERT backbone's "
                "pre-training on Wikipedia text and tables may still include public financial text, and that exposure was not "
                "checked and cannot be ruled out, so the frozen scores are not a guarantee of out-of-distribution behaviour. "
                "Both sources come from financial reports of large US companies; the two sources may share companies, and FinQA "
                "companies recur across years, so the split keeps each **FinQA report** (one company's filing for one year) and "
                "each **TAT-QA table** on one side — a company can still appear in two splits through different years' reports.\n\n"
                "**What is not covered.** No question in the sample is a `COUNT` question, so `COUNT` accuracy is not measured; "
                "the earlier WikiSQL `max`/`min` categories are gone too."
            ),
        ]
    },
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch digest-pinned labelled table-question sets, convert their answers to gold cells, validate "
        "them and split them by source report and table without leakage; answer a synthetic table through the public API and read the "
        "cell/operator/numeric contract correctly (fixed threshold, argmax, pipeline arithmetic, no abstention); measure the "
        "frozen model's value-equality and strict-operator accuracy (co-primary) plus operator and cell accuracy per question "
        "type beside two non-neural baselines; run a "
        "bounded fine-tuning with the model's own weak-supervision loss, explicit hyperparameters and validation-based epoch "
        "selection; evaluate on a table-disjoint test split; re-answer the synthetic table with the adapted model; and "
        "export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "conversational or multi-turn table QA (the SQA setting), tables with more than `MAX_ROWS` rows or `MAX_COLUMNS` "
        "columns, non-English tables, text-to-SQL or arbitrary computation beyond the four aggregation operators, the "
        "`MAX`/`MIN` operators, **`COUNT` accuracy** (the default sample has no `COUNT` questions in this revision), "
        "questions that need text outside the table (TAT-QA and FinQA questions of that kind are filtered out), arithmetic "
        "beyond a sum or an average (differences, ratios, percentage changes), fine-tuning of the embeddings or of any encoder "
        "block but the last ones, evaluation on TAT-QA, FinQA or WikiTableQuestions proper (only one seeded 365-question "
        "sample is scored here), and any claim that three question types on financial-report tables stand in for your "
        "tables. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with TAPAS or fine-tuning. The notebook explains cell selection, the aggregation operator, the two answer rules, weak supervision and the adapter where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter); a GPU runtime is strongly recommended. Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it. The default path runs on CPU (float32) and uses CUDA automatically when available (also float32). CPU is slow: TAPAS-large takes on the order of a second per question on a workstation CPU, and fine-tuning repeats that for every training and validation question per epoch, so a CPU run takes well over half an hour and a 2-vCPU hosted runtime a multiple of that. No hosted timing of this revision is recorded yet. The pinned `torch==2.14.0` install and the 1.35 GB checkpoint are the large downloads of the run; the four data files add about 116 MB.",
        "- **Knowledge:** basic Python; what a sigmoid threshold and an argmax are and why neither is a calibrated probability; that an aggregation over selected cells is arithmetic the pipeline performs, not a model output; what an accuracy measures and why one seeded split gives no dispersion.",
        "- **Data contract:** records are `{id, table, question, answer}` with an optional `category` — `table` is `{column: [cells]}` with every header and cell a str (at most `MAX_ROWS` = 64 rows, `MAX_COLUMNS` = 32 columns, `MAX_CELL_CHARS` = 200 characters per cell), `question` a non-empty str of at most `MAX_QUERY_CHARS` = 500 characters, `answer` = `{aggregation: NONE|SUM|AVERAGE|COUNT, coordinates: [[row, col], ...], denotation: [cell, ...] | number}` where a `NONE` denotation is the selected cells and an operator denotation is the number the operator produces from them (validated). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a training split needs 8..20,000 records and validation and test at least one each, so with the default 15 % + 20 % table hold-out the effective BYOD minimum is **12 questions — one per table over 12 tables, or two per table over 6** (`min_byod_records()` computes it); splitting is by table (normalised content) so no table lands in two splits; a table whose gold cells fall past the `MAX_TOKENS` = 512 truncation point is refused at training time. BYOD accepts one JSONL file (one record per line, UTF-8 with or without a byte-order mark; a broken line is refused with its line number) or a JSON list in that shape. Numbers parse as plain digits with an optional sign, decimal point and thousands commas; `−12`, `$12`, `(12)`, `1 234` and `1.2e3` do not, and such cells are reported as `unparsed_cells` (TPQ-S2). The default sample's cells are rewritten into the parseable form when the sample is built (`clean_cell`); BYOD cells are used as given.",
        "- **Validation is structural, not semantic:** every table, question and answer is checked for shape and arithmetic consistency, but nothing checks that an answer is right — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path fetches four JSON files, each pinned by byte count and SHA-256 in the carried `samples.py` (`CORPORA`) and refused on any mismatch: `tatqa_dataset_train.json` from the Hugging Face Hub dataset `next-tat/TAT-QA` at revision `c96247f5077e…` (12.8 MB; CC BY 4.0), and `dataset/train.json`, `dev.json` and `test.json` from `raw.githubusercontent.com/czyssrs/FinQA` at commit `0f16e2867bef…` (78.2, 11.0 and 14.4 MB; MIT, attributed as CC BY 4.0 per its Hub card). Nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Financial-table sample and split\n\n"
                "`fetch_corpus` returns each pinned file from the cache under `weights/table-qa-corpora/` or its source — every "
                "cached or fetched copy is refused on a byte-count or SHA-256 mismatch. `sample_candidates` converts them: "
                "`tatqa_candidates` keeps TAT-QA's table-span questions whose answer cells each occur once in a table that fits the "
                "sample ceilings (at most 20 rows, 12 columns and 1,000 characters, so nothing is truncated at 512 tokens), and "
                "`finqa_candidates` keeps FinQA's table-only sum and average programmes whose every term matches exactly one cell "
                "and whose pipeline arithmetic reproduces FinQA's executed answer. `build_sample_dataset` shuffles each source's "
                "**groups** — a FinQA report (one company's filing for one year) or a TAT-QA table — assigns each group to one "
                "split first, then draws 80 lookups, 60 sums and 60 averages for training and 20 / 35 per type for validation / "
                "test (200 / 60 / 105). There is no `count` type: COUNT is not exercised in this revision. `validate_dataset` checks "
                "every record against the contract — including that each operator denotation is what the pipeline's arithmetic "
                "produces from the gold cells — `check_split_disjoint` asserts no table (by normalised content) is shared, and the "
                "training split is written to `outputs/{stem}_train.jsonl` in the BYOD shape.\n\n"
                "Look for: the filter funnel (questions per source before and after the table ceilings and the gold-cell match), "
                "the three categories with 80 / 60 / 60 in training and 20 / 35 elsewhere, three digests, and four refusal probes — "
                "a duplicate id, a coordinate outside the table, an operator denotation that disagrees with its cells, and a dataset "
                "too small to use — each rejected before the model does anything.\n\n"
                "*Evaluation practice.* **Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (a JSONL "
                "file in this runtime — this works on Colab, Kaggle and Jupyter) or leave `BYOD_PATH` empty to upload exactly one "
                "file through the Colab dialog; then choose **Run after** from this cell. This cell first puts the model back to "
                "the pinned base, so Section 6's frozen row and Section 7's epoch 0 are the untouched checkpoint. The effective "
                "minimum is 12 questions (one per table over 12 tables, or two per table over 6).\n\n"
                "**Predict before running:** several questions share a table, and one company files a report every year. Why does "
                "the split keep every question of a table — and every table of a report — on one side?"
            ),
            "code": (
                "import csv\n"
                "import hashlib\n"
                "import json\n"
                "import shutil\n"
                "import time\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "# A re-run after Section 7 (BYOD, or a new split): Section 6's frozen row and Section 7's epoch 0 must be the pinned base.\n"
                "had_adapter = pipe.adapter is not None\n"
                "restored_tensors = pipe.restore_base()\n"
                "if had_adapter or restored_tensors:\n"
                "    print({{'restored_pinned_base': len(restored_tensors), 'note': 'the fine-tuned encoder blocks and heads were put back to the checkpoint; Sections 5-7 start from it again'}})\n"
                "class ByodRefusal(ValueError):\n"
                "    \"\"\"A BYOD input this notebook refuses: shown as one message, without a traceback.\"\"\"\n"
                "    dimer_refusal = True\n\n"
                "    def _render_traceback_(self):\n"
                "        return [str(self)]\n\n\n"
                "def load_byod():\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_path.is_file():\n"
                "            raise ByodRefusal(f'BYOD_PATH {{BYOD_PATH!r}} is not a file (relative paths start at {{Path.cwd()}}): give one JSONL file of records.')\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise ByodRefusal('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the JSONL file in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ByodRefusal(f'Upload exactly one JSONL file (received {{len(uploaded)}}; a cancelled dialog sends none).')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        byod_path = Path('work') / 'byod.jsonl'\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    byod_splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    return byod_splits, 'BYOD (' + file_name + ')', {{'byod': len(records), 'effective_minimum': min_byod_records()['total']}}\n\n\n"
                "def refuse_byod(exc):\n"
                "    # Results of an earlier run must not be read as this file's: remove them before stopping.\n"
                "    stale = sorted(p.name for p in Path('outputs').glob('{stem}_*'))\n"
                "    for p in Path('outputs').glob('{stem}_*'):\n"
                "        shutil.rmtree(p) if p.is_dir() else p.unlink()\n"
                "    for name in ('splits', 'train_records', 'val_records', 'test_records', 'frozen_test', 'adapted_test', 'adapt_result'):\n"
                "        globals().pop(name, None)\n"
                "    note = f' Removed {{len(stale)}} earlier output(s) so they are not mistaken for this file\\'s results.' if stale else ''\n"
                "    raise ByodRefusal(f'BYOD refused: {{exc}} Fix the file and run this cell again.{{note}}') from None\n\n\n"
                "if USE_BYOD:\n"
                "    try:\n"
                "        splits, data_source, raw_rows = load_byod()\n"
                "    except (ByodRefusal, ValueError, TypeError) as exc:\n"
                "        refuse_byod(exc)\n"
                "    if len(splits['test']) < 20:\n"
                "        print({{'caution': f\"only {{len(splits['test'])}} held-out test questions: accuracy moves in large steps and carries no dispersion estimate; add questions before reading it\"}})\n"
                "else:\n"
                "    corpora = {{key: fetch_corpus(key, cache_dir='weights/table-qa-corpora') for key in CORPORA}}\n"
                "    funnel = {{}}\n"
                "    candidates = sample_candidates(corpora, funnel)\n"
                "    splits = build_sample_dataset(candidates, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} — {{CORPUS_LICENSE}}'\n"
                "    raw_rows = {{'file_bytes': {{key: len(data) for key, data in corpora.items()}}, 'filter_funnel': funnel, 'candidates': candidate_summary(candidates)}}\n"
                "# The training split must hold MIN_RECORDS; validation and test only need one question each (split_dataset checks that).\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_jsonl(train_records, 'outputs/{stem}_train.jsonl')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'tables': manifest['n_tables'], 'categories': manifest['category_counts'], 'aggregations': manifest['aggregation_counts'], 'table_shape': manifest['table_shape'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{'id': example['id'], 'category': example.get('category'), 'question': example['question'], 'source': example.get('source'), 'program': example.get('program'), 'answer': example['answer'], 'columns': list(example['table']), 'rows': len(next(iter(example['table'].values())))}}}})\n\n"
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
                '**What to notice:** the filter funnel, 200 / 60 / 105 questions, the per-split category counts (no `count`), `shared_tables` of zero and the four refusals.\n\n<details><summary>Check your reasoning</summary>Questions about one table share its layout and often its answers; split by question, the model could be tested on a table it was trained on and the score would measure memory of that table. A company\'s report for one year repeats many of the same rows across its tables, so FinQA is split by report, not just by table. The test questions then sit on tables unseen by this notebook\'s fine-tuning (the checkpoint\'s own pre-training exposure is discussed in the Data context).</details>'
            ),
        },
        {
            "md": (
                "## 5. Answer the synthetic table through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a 4-row × 3-column table of "
                "Philippine cities authored in code (the model card's smoke table, population cells with thousands "
                "separators) and three questions — a **lookup**, a **COUNT** and a **SUM** — with the answers the author "
                "expects; a different table family from the financial sample, and questions the adapted model will answer again in "
                "Section 9. `validate_inputs` applies exactly the checks `answer` applies (`MAX_ROWS`, `MAX_COLUMNS`, "
                "`MAX_CELL_CHARS`, `MAX_QUERY_CHARS`; `MAX_TOKENS` is applied by the tokenizer and *truncates*, reported by "
                "`answer`) and returns an input manifest; a table carrying a numeric (non-string) cell is validated too and "
                "its rejection recorded as a finding. `answer(table, query)` returns `cells`, `coordinates`, `aggregation`, "
                "`numeric_answer` with `numeric_answer_source`, `unparsed_cells`, `aggregation_logits`, `n_tokens`, "
                "`tokens_before_truncation`, `rows_kept`, `truncated` and the decision rule; the per-grid `evaluation_report` "
                "on three authored questions is `sample-sanity` — plumbing evidence, not a measurement; whether the model is "
                "*right* is what Section 6 measures on 105 questions.\n\n"
                "*Core concept.* TAPAS never writes a number: it selects cells and picks an operator, and the pipeline does the "
                "arithmetic. The cell threshold and the operator argmax are decisions, not calibrated probabilities.\n\n"
                "**Predict before running:** the population cells carry thousands separators (`1,846,513`). Will the SUM question "
                "get the right total?"
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
                "**What to notice:** each question's operator, its cells, `numeric_answer` and `unparsed_cells`.\n\n<details><summary>Check your reasoning</summary>If the model selects the Manila and Davao cells and picks `SUM`, yes: the pipeline parses thousands commas, so the total is exact. A wrong answer here is a cell or operator choice, not arithmetic — read `aggregation` and `cells` first. Three authored questions are plumbing evidence; Section 6 measures accuracy on 105.</details>"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test questions\n\n"
                "Three systems frame the adaptation, each scored by `denotation_metrics` (carried in `metrics.py`) on the "
                "pipeline's **final answer** — the selected cells under `NONE`, otherwise the number the predicted operator "
                "produces. Two rules are **co-primary**: **value-equality accuracy** (`value_accuracy`: the final answer's value "
                "equals the gold value whatever operator produced it — a `NONE` answer whose one cell equals a sum's gold number "
                "counts as right) and **strict-operator accuracy** (`strict_accuracy`: the operator must also be the gold one). "
                "Neither is a headline: value-equality asks *is the answer right*, strict-operator asks *was it reached the intended "
                "way*, and a gap between them is the operator error this notebook is about. **Aggregation accuracy** (the "
                "predicted operator equals the record's) and **cell accuracy** (the selected coordinates equal the record's as a "
                "set) are diagnostics. All four are reported overall and per question type. The **first-cell "
                "floor** answers every question with the first cell of the first column and `NONE`. The **keyword lookup** "
                "picks the column whose header shares the most words with the question, the rows whose other cells appear "
                "verbatim in it, and an operator from the question's wording (`how many` → `COUNT`, `total` → `SUM`, "
                "`average` → `AVERAGE`) — a table-QA system that never sees a token embedding. The **frozen model** is scored "
                "by `pipe.evaluate`, which answers every record through `answer` and scores the results. **What to look for:** "
                "the frozen model against both baselines on each measure *separately*, the gap between value-equality and "
                "strict-operator, and the operator column "
                "per question type.\n\n"
                "*Evaluation practice.* The cell records a **verdict** on the ordering instead of asserting it: on your questions "
                "the keyword lookup can tie or beat the frozen model (questions that quote cell values verbatim), and that is a "
                "finding — the notebook continues to fine-tuning.\n\n"
                "**Predict before running:** the keyword lookup picks an operator from words like *how many* and *total*. Will it "
                "beat the frozen model on **operator** accuracy?"
            ),
            "code": (
                "METRICS = ('value_accuracy', 'strict_accuracy', 'aggregation_accuracy', 'cell_accuracy')\n"
                "baseline_first = first_cell_baseline(test_records)\n"
                "baseline_keyword = keyword_lookup_baseline(test_records)\n"
                "print({{'first_cell_baseline': {{k: round(baseline_first[k], 3) for k in METRICS}}, 'n': baseline_first['n'], 'note': baseline_first['baseline']}})\n"
                "print({{'keyword_lookup_baseline': {{k: round(baseline_keyword[k], 3) for k in METRICS}}, 'by_category': {{c: {{'value': round(v['value_accuracy'], 2), 'strict': round(v['strict_accuracy'], 2)}} for c, v in baseline_keyword['per_category'].items()}}, 'note': baseline_keyword['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'truncated': frozen_test['truncated'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'value': round(v['value_accuracy'], 2), 'strict': round(v['strict_accuracy'], 2), 'aggregation': round(v['aggregation_accuracy'], 2), 'cells': round(v['cell_accuracy'], 2)}} for c, v in frozen_test['per_category'].items()}}\n"
                "print({{'by_category_frozen': frozen_fields}})\n"
                "# A reported verdict, not an assertion: on your questions a baseline may tie or beat the frozen model, and that is a finding.\n"
                "frozen_verdict = {{metric: ('frozen above both baselines' if frozen_test[metric] > max(baseline_keyword[metric], baseline_first[metric]) else 'a baseline ties or beats the frozen model') for metric in METRICS}}\n"
                "print({{'frozen_vs_baselines': frozen_verdict}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the three measures per system, and `aggregation` per type in `by_category_frozen`.\n\n<details><summary>Check your reasoning</summary>It can: FinQA questions say *total* and *average* outright, so the keyword rule's operator column can be strong on `sum` and `average` while its cell choice — and therefore both answer rules — stays weak. The frozen model usually finds better cells; whether it also picks `SUM` and `AVERAGE` when asked is what the strict-operator column shows. Read the two answer rules side by side: where value-equality is well above strict-operator, the model reached the right number the wrong way (for example a `NONE` lookup of a cell that happens to equal the sum). No hosted numbers for this revision are recorded yet; read yours.</details>"
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
                "every epoch is scored on the 60 validation questions and the epoch with the highest validation **score** — "
                "the mean of value-equality, aggregation and cell accuracy, a steadier selector than one answer rule alone on "
                "60 questions — is kept. Epoch selection reads the validation split only; the test split is not consulted.\n\n"
                "**What to look for:** the training loss and the validation aggregation accuracy per epoch, and which epoch the "
                "score selects. (What did not work in earlier revisions is listed at the end.)\n\n"
                "*Core concept.* Every call to `pipe.adapt` starts from the **pinned base**: tensors an earlier call (or an "
                "artifact) changed are put back first, so epoch 0 is always the frozen model and re-running Sections 7–8 with a "
                "changed field repeats the comparison validly. To compare a change side by side without replacing the default "
                "exports, use Section 10.\n\n"
                "**Predict before running:** which operator types will fine-tuning help, and could it *hurt* one?"
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
                "settings = {{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'trainable_layers': TRAINABLE_LAYERS}}\n"
                "if settings != {{'epochs': 4, 'lr': 5e-5, 'batch_size': 8, 'trainable_layers': 2}}:\n"
                "    print({{'note': 'changed settings: this run starts again from the pinned base and replaces the default results of Sections 8-9; Section 10 compares a change side by side instead', 'settings': settings}})\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_layers=TRAINABLE_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the epoch-0 row (the frozen model on validation), the jump in `val_aggregation_accuracy`, and `best_epoch`.\n\n<details><summary>Check your reasoning</summary>It helps the types whose operator the frozen model gets wrong — on this sample, `sum` and `average` questions the WTQ head answers with `NONE` — and it can hurt a type it already handled, such as lookups. Section 8 shows per type which happened in your run. The training loss keeps falling while the validation score picks an earlier or later epoch: the score, not the loss, decides.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "In this notebook's fine-tuning, the test questions' tables (and their FinQA reports) were never used for training "
                "or epoch selection, and no table appears in two splits. The adapted model is scored exactly as the frozen model "
                "was in Section 6, the four systems are put side by side on the four measures, and the per-type breakdown is "
                "repeated. Read it in this order: the two co-primary answer rules together — **value-equality** and "
                "**strict-operator** — then **aggregation accuracy** (what the adaptation teaches) and **cell accuracy**, then the "
                "per-type rows — both answer rules and the **operator accuracy per type**, because an overall operator gain can "
                "hide a type whose operator the model *unlearned*. On 35 questions per type, one question is about three points; "
                "on 105, the answer rules move by single questions, in either direction across runs. The "
                "cell records verdicts — whether each measure rose — instead of asserting them, so "
                "a set where the frozen model already picks every operator (aggregation 1.0) still completes, exports and "
                "writes the result. One seeded split gives **no dispersion estimate**; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a gain on three question types "
                "over financial-report tables says nothing about your tables until you measure them. COUNT is not measured."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "adapted_fields = {{c: {{'n': v['n'], 'value': round(v['value_accuracy'], 2), 'strict': round(v['strict_accuracy'], 2), 'aggregation': round(v['aggregation_accuracy'], 2), 'cells': round(v['cell_accuracy'], 2)}} for c, v in adapted_test['per_category'].items()}}\n"
                "comparison = {{metric: {{'first_cell': round(baseline_first[metric], 3), 'keyword': round(baseline_keyword[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['by_category'] = {{c: {{'n': frozen_fields[c]['n'], 'frozen_value': frozen_fields[c]['value'], 'adapted_value': adapted_fields[c]['value'], 'frozen_strict': frozen_fields[c]['strict'], 'adapted_strict': adapted_fields[c]['strict'], 'frozen_aggregation': frozen_fields[c]['aggregation'], 'adapted_aggregation': adapted_fields[c]['aggregation']}} for c in frozen_fields}}\n"
                "def direction(new, old):\n"
                "    return 'improved' if new > old else ('no gain' if new == old else 'worse')\n"
                "# Reported verdicts, not assertions: a measure that does not rise is a result to record, and export and reload still run.\n"
                "comparison['verdicts'] = {{'frozen_vs_baselines': frozen_verdict, **{{f'adapted_vs_frozen_{{metric}}': direction(adapted_test[metric], frozen_test[metric]) for metric in METRICS}}}}\n"
                "comparison['operator_regressions'] = sorted(c for c in frozen_fields if adapted_fields[c]['aggregation'] < frozen_fields[c]['aggregation'])\n"
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
                "print({{'verdicts': comparison['verdicts'], 'operator_regressions': comparison['operator_regressions']}})\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the overall `delta_vs_frozen`, then `by_category` — answer and operator accuracy per type — and `operator_regressions`.\n\n<details><summary>Check your reasoning</summary>Look first at the gap between the two answer rules: a fine-tune that teaches the operator should raise strict-operator accuracy more than value-equality, because a frozen model that answers a sum with a `NONE` lookup of the right total is already right by value. Then read `operator_regressions`: on the previous revision (blob `f8e0d361`, WikiSQL sample; Colab CLI T4 run of 2026-10-10 at `1f8ba58`), the same recipe raised aggregation accuracy from 0.567 to 0.807 while `sum` operator accuracy fell from 0.88 to 0.56 — numbers quoted from that run on different data, not re-run here. The lesson carries over: an overall operator gain can hide a regression on one type, and single-question movements are noise.</details>"
            ),
        },
        {
            "md": (
                "## 9. Re-answer the synthetic table, export the adapter and reload it\n\n"
                "The three city-table questions from Section 5 are answered again by the adapted model — a table family the "
                "adaptation never saw, so this is a small look at what the adaptation did *outside* its corpus (a changed answer "
                "here is a finding to record, not a failure) — "
                "reported with the per-grid `evaluation_report` (`sample-sanity`), and both answer sets are written to "
                "`outputs/{stem}_answers.csv`.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last two encoder blocks and the three heads, about "
                "100 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the training "
                "configuration and the epoch history (OUT8). `TAPASTableQAPipeline.from_artifact` re-verifies the base "
                "snapshot, checks the artifact manifest, its digest and its exact tensor set **before** deserialising, refuses "
                "any tensor outside the encoder blocks and heads, and overlays the tensors onto a freshly loaded base — a new "
                "object from files, not the in-memory model (VER2). The cell asserts identical cells, operators and "
                "aggregation logits on eight test questions (VER4) — a contract check, so it stays a hard check.\n\n"
                "**Predict before running:** the adapted model learned on financial-report tables. Will it answer the city table's SUM "
                "question the same way as the frozen model?"
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
                "    'corpus': {{'name': CORPUS_NAME, 'files': CORPUS_PINS, 'license': CORPUS_LICENSE, 'sample_digest': SAMPLE_DIGEST, 'categories': list(SAMPLE_CATEGORIES), 'count_exercised': False}},\n"
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
        {
            "md": (
                "**What to notice:** the frozen and adapted rows for each city question in `outputs/{stem}_answers.csv`, and the reload parity line.\n\n<details><summary>Check your reasoning</summary>Not necessarily: the city table's questions include a `COUNT`, which this sample never trains, and a sum over population cells, so a changed answer would be a finding about change outside the corpus — from three questions, not a measurement. Reload parity is a contract check: the cell asserts eight of eight identical answers, so a completed run means it held.</details>"
            ),
        },
        {
            "md": (
                "## 10. Change one thing: train four encoder blocks (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, change **one** field — by default four encoder blocks train "
                "instead of two — and run this cell after Sections 4–9. The experiment loads its **own** pipeline from the "
                "verified snapshot, so it starts from the checkpoint and never touches the default `pipe`; it writes only to "
                "`outputs/{stem}_experiment/`, prints the default and the changed run side by side — overall and operator "
                "accuracy per type — and checks that the default exports (adapter, evaluation report, result) are byte-identical "
                "afterwards. About as long as Section 7 again.\n\n"
                "It **compares the two recipes on the test questions for illustration, not recipe selection**: the side-by-side is on the test "
                "questions, and the four-block run reuses a recipe (learning rate, epochs) chosen for two blocks, so neither run "
                "is tuned for the comparison. Do not pick a setting from it; tune on validation if you need to choose.\n\n"
                "**Predict:** with twice the trainable blocks, will operator accuracy rise further, and what happens to lookups?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_TRAINABLE_LAYERS = 4  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': Path('outputs/{stem}_evaluation_report.json'), 'result': Path('outputs/{stem}_result.json')}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own pipeline from the verified snapshot: the experiment starts from the checkpoint and the default pipe is untouched.\n"
                "    experiment_pipe = TAPASTableQAPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "    experiment_result = experiment_pipe.adapt(train_records, val_records, epochs=EXPERIMENT_EPOCHS, lr=EXPERIMENT_LEARNING_RATE, batch_size=BATCH_SIZE, trainable_layers=EXPERIMENT_TRAINABLE_LAYERS, progress=report)\n"
                "    experiment_test = experiment_pipe.evaluate(test_records)\n"
                "    side_by_side = {{\n"
                "        'settings': {{'default': {{'trainable_layers': adapt_result['trainable_layers'], 'epochs': adapt_result['epochs'], 'lr': adapt_result['lr']}}, 'experiment': {{'trainable_layers': EXPERIMENT_TRAINABLE_LAYERS, 'epochs': EXPERIMENT_EPOCHS, 'lr': EXPERIMENT_LEARNING_RATE}}}},\n"
                "        'best_epoch': {{'default': adapt_result['best_epoch'], 'experiment': experiment_result['best_epoch']}},\n"
                "        'epoch_0_val_score': {{'default': (adapt_result['history'][0]['val'] or {{}}).get('score'), 'experiment': (experiment_result['history'][0]['val'] or {{}}).get('score')}},\n"
                "        'test': {{metric: {{'frozen': round(frozen_test[metric], 3), 'default': round(adapted_test[metric], 3), 'experiment': round(experiment_test[metric], 3)}} for metric in METRICS}},\n"
                "        'operator_by_category': {{c: {{'frozen': frozen_fields[c]['aggregation'], 'default': adapted_fields[c]['aggregation'], 'experiment': round(v['aggregation_accuracy'], 2)}} for c, v in experiment_test['per_category'].items() if c in frozen_fields}},\n"
                "        'trainable_parameters': {{'default': adapt_result['n_trainable'], 'experiment': experiment_result['n_trainable']}},\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    with open(experiment_dir / 'experiment_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'history': experiment_result['history']}}, handle, indent=2, ensure_ascii=False, default=str)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': str(experiment_dir)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare `epoch_0_val_score` (it must be equal: both runs start from the checkpoint), the "
                "`test` rows and `operator_by_category`.\n\n"
                "<details><summary>Check your reasoning</summary>No experiment run of this revision is recorded. On the previous "
                "revision's WikiSQL sample (build record of blob `11134191` at `b0ebfe4`, workstation CPU; not re-run here), four "
                "blocks at the same rate raised operator accuracy about as much as two while lookups fell. That is an observation "
                "from one untuned run, not a mechanism. On 35 questions per type, read a difference of one or two questions as "
                "noise, and look for the trade between operator and lookup rows rather than the overall number.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "This revision has no hosted run yet, so this section states what the run measures and how to read it rather than "
        "quoting results. The frozen WTQ checkpoint meets financial-report tables it was not fine-tuned on; the two "
        "co-primary rules separate *is the answer right* (value-equality) from *was it reached with the intended operator* "
        "(strict-operator), and the fine-tuning of the last two encoder blocks and the heads on 200 questions is read on "
        "both, per question type, against two non-neural baselines and the frozen model. The claim the notebook supports is "
        "that the adaptation contract works end to end on a real labelled table-question set and that its numbers are read "
        "on several measures rather than one; the 100 MB adapter reloads to identical answers (asserted).\n\n"
        "The test split is 105 questions from one seeded draw of one sample, the validation split that picks the epoch is 60, "
        "and the answer rules move in steps of one question. So a gain here says the contract works, not that the adapted "
        "model is better on your tables, that a sigmoid threshold or an argmax is a probability, or that the pipeline's "
        "arithmetic is right when the cells are wrong — it still returns some cells and some operator for every question, "
        "and it is confidently wrong when it is wrong. **COUNT is not exercised:** no question in the sample needs it, so "
        "nothing here says whether the adaptation helps or harms `COUNT`. Fine-tuning on a narrow set can also erode the "
        "model elsewhere; the city table re-answered in Section 9 is three questions of evidence about that, not a "
        "measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the first-cell floor, the keyword lookup and the frozen "
        "model's score on *your* questions are the numbers to read before any adapted one, per question type. **Leakage:** "
        "keep every table in one split (the contract splits by normalised table content) and split by source document when "
        "your tables come from few documents, as the default sample does by FinQA report. **Supervision:** the gold cells of a lookup and the number of an operator question "
        "are the two shapes the weak-supervision loss understands; a table whose gold cells lie past the 512-token truncation "
        "point is refused at training time, and a mislabelled answer is learned without complaint.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify real labelled table-question sets, "
        "validate the demonstrated dataset contract without leakage, execute the inference contract and a bounded fine-tuning, "
        "evaluate against two trivial baselines and the frozen model on a table-disjoint split, and emit the shown "
        "machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark superiority, "
        "accuracy on any other domain or table shape, `COUNT` accuracy, a usable threshold, or production fitness.\n\n"
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 trains four encoder "
        "blocks in its own pipeline and prints it beside the default run — change `EXPERIMENT_TRAINABLE_LAYERS`, "
        "`EXPERIMENT_EPOCHS` or `EXPERIMENT_LEARNING_RATE` (for example `2e-5`) there and run that cell again. Changing "
        "`EPOCHS`, `LEARNING_RATE` or `TRAINABLE_LAYERS` and choosing **Run after** from Section 7 also starts from the pinned "
        "base — every `adapt` puts it back first — but replaces the default results and exports. BYOD: `USE_BYOD` and "
        "`BYOD_PATH` in Section 4, then **Run after** from Section 4, and read the two baselines before the adapted number.\n\n"
        "## What did not work\n\n"
        "- **WikiSQL as the default data** (revisions up to `28d7bba`). Tried: a WikiSQL validation-shard sample with "
        "`MAX`/`MIN` mapped to `NONE` lookups. Observed: the checkpoint was fine-tuned on WikiSQL itself (its chain is SQA → "
        "WikiSQL → WTQ), so the frozen scores were not measured on shifted data. Rejected for TAT-QA and FinQA (review "
        "TQA-M2, owner decision 2026-10-15).\n"
        "- **One answer rule as the headline** (revisions up to `28d7bba`). Tried: a single denotation accuracy that compared "
        "list-gold questions on cells only. Observed: it ignored the predicted operator and hid the fine-tune's operator gain "
        "(review TQA-M1). Replaced by value-equality and strict-operator, reported together.\n"
        "- **TAT-QA for `COUNT`.** Tried while choosing the data: TAT-QA's count questions. Observed: 67 usable questions on "
        "64 tables after the gold-cell match, fewer than a split needs; FinQA has no count programme. Rejected: `COUNT` is "
        "left unexercised in this revision rather than measured on too few questions.\n"
        "- **Four trainable blocks, or a training draw with as many operator questions as lookups** (build record of blob "
        "`11134191` at `b0ebfe4`, workstation CPU, WikiSQL sample; not re-run here). Observed: operator accuracy rose about "
        "as much as with two blocks, but lookups fell. Rejected for the default recipe; Section 10 lets you try four blocks.\n\n"
        "## Troubleshooting\n\n"
'- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering `files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and choose **Run all**; leave the optional experiment off on a small runtime.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again; the snapshot comes from `huggingface.co`.\n- **Section 4 cannot fetch a data file, or one fails its digest** — the default path needs `huggingface.co` (TAT-QA) and `raw.githubusercontent.com` (FinQA); run Section 4 again, and delete `weights/table-qa-corpora/` if a cached copy is corrupt.\n- **Section 7 is very slow** — TAPAS-large on a 2-vCPU CPU runtime takes a long time; choose a GPU runtime and **Run all** again.\n- **CUDA out of memory** — lower `BATCH_SIZE` in Section 7 and run Sections 7–9 again, or restart the session and choose **Run all**; leave Section 10 off on a small GPU.\n- **BYOD refusals** are one line starting `BYOD refused:` with no traceback; earlier outputs are removed so they cannot be mistaken for the results of your file. Fix the file and run Section 4 again.\n- **BYOD: "BYOD_PATH … is not a file"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the JSONL file in the runtime (or attach it as a dataset) and set `BYOD_PATH` to its path.\n- **BYOD: "Upload exactly one JSONL file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **BYOD: "… line N: not valid JSON …"** — that line of your file is broken; each line must be one record object.\n- **BYOD: "… distinct tables are too few" or "the split leaves … training questions"** — add tables or questions; the message names the minimum.\n- **BYOD: a record is refused by `validate_dataset`** — the message names the record and the rule (an operator denotation must be the number its cells give).\n'
        "## Glossary\n\n"
        "- **Cell selection** — the cells whose mean token probability exceeds `CELL_THRESHOLD` (0.5).\n"
        "- **Aggregation operator** — `NONE`, `SUM`, `AVERAGE` or `COUNT`, an argmax over the aggregation head.\n"
        "- **Numeric answer** — the pipeline's arithmetic over the selected cell strings, never a model output.\n"
        "- **Final answer** — the selected cells under `NONE`, otherwise the number the predicted operator produces.\n"
        "- **Value-equality accuracy** — the fraction of questions whose final answer has the gold value, whatever operator "
        "produced it (numbers to 1e-6 relative; a single numeric cell compares as a number).\n"
        "- **Strict-operator accuracy** — the fraction whose operator is the gold one *and* whose value is right. Co-primary "
        "with value-equality; neither is the headline.\n"
        "- **Aggregation accuracy / cell accuracy** — the operator, or the selected coordinates, equal the record's.\n"
        "- **Weak supervision** — an operator question is trained only against its number (`float_answer`); a lookup labels "
        "its gold cells.\n"
        "- **Gold-cell conversion** — finding the table cells behind a TAT-QA span answer or each term of a FinQA programme; "
        "a question whose cells are ambiguous is dropped.\n"
        "- **Report-disjoint split** — every table of a FinQA report (one company's filing for one year) on one side.\n"
        "- **First-cell floor / keyword lookup** — two non-neural baselines that frame the model's numbers.\n"
        "- **Table-disjoint split** — every question of a table on one side of the split.\n"
        "- **Adapter / reload parity** — the trained tensors only (safetensors) overlaid on the pinned base; the reloaded "
        "pipeline gives identical answers.\n"
        "- **BYOD** — bring your own data: your labelled table questions through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The task was ___ on ___ test questions over ___ tables.\n"
        "- The keyword lookup scored value-equality ___ / strict-operator ___ / operator ___; the frozen model ___ / ___ / ___.\n"
        "- After fine-tuning (epoch ___ selected): value-equality ___, strict-operator ___, operator ___, cells ___.\n"
        "- Types that gained: ___; types that lost (answer or operator): ___.\n"
        "- What I would need before claiming fine-tuning helps: ___ (for example more test questions, several seeds).\n\n"
        "## Repository documents\n\n"
        "- Repository README: https://github.com/kurtvalcorza/tapas-table-question-answering-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/tapas-table-question-answering-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/tapas-table-question-answering-pipeline/blob/main/docs/WEIGHTS.md\n\n"
        "## References\n\n"
        "[1] J. Herzig, P. K. Nowak, T. Müller, F. Piccinno and J. Eisenschlos, \"TAPAS: Weakly supervised table parsing via pre-training,\" in *Proc. ACL*, 2020. [Online]. Available: https://arxiv.org/abs/2004.02349\n\n"
        "[2] J. Eisenschlos, S. Krichene and T. Müller, \"Understanding tables with intermediate pre-training,\" in *Findings of EMNLP*, 2020. [Online]. Available: https://arxiv.org/abs/2010.00571\n\n"
        "[3] Google, \"google/tapas-large-finetuned-wtq,\" Hugging Face model card. [Online]. Available: https://huggingface.co/{MODEL_ID} ; code: https://github.com/google-research/tapas\n\n"
        "[4] P. Pasupat and P. Liang, \"Compositional semantic parsing on semi-structured tables,\" in *Proc. ACL*, 2015. [Online]. Available: https://arxiv.org/abs/1508.00305\n\n"
        "[5] F. Zhu *et al.*, \"TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance,\" in *Proc. ACL*, 2021. [Online]. Available: https://aclanthology.org/2021.acl-long.254/ ; data (CC BY 4.0): https://huggingface.co/datasets/next-tat/TAT-QA\n\n"
        "[6] Z. Chen *et al.*, \"FinQA: A dataset of numerical reasoning over financial data,\" in *Proc. EMNLP*, 2021. [Online]. Available: https://arxiv.org/abs/2109.00122 ; data (MIT; CC BY 4.0 on its Hub card): https://github.com/czyssrs/FinQA\n\n"
        "[7] V. Zhong, C. Xiong and R. Socher, \"Seq2SQL: Generating structured queries from natural language using reinforcement learning,\" arXiv:1709.00103, 2017 (WikiSQL, the earlier default data). [Online]. Available: https://arxiv.org/abs/1709.00103\n\n"
        "[8] K. Valcorza, \"tapas-table-question-answering-pipeline: README, model card and weight provenance,\" GitHub repository. [Online]. Available: https://github.com/kurtvalcorza/tapas-table-question-answering-pipeline\n\n"
        "[9] DIMER, \"DIMER Notebook Specification 2.5\" and \"Model Card Specification 1.1,\" fleet specifications in the ml-worker repository. [Online]. Available: https://github.com/kurtvalcorza/ml-worker/tree/main/integrations/dimer/fleet-specs"
    ),
}
