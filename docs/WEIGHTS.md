# Weight provenance and DIMER hosting

- Upstream: `google/tapas-large-finetuned-wtq`
- Immutable revision: `f58317ab2577d17647d9acafa790c744a0388b30`
- Weight format: SafeTensors (`model.safetensors`, 1346985282 bytes, SHA-256 `149247e13732c222ba621e0c4e7b90ba260869b36adcbe115dc872c2c98bccf0`)
- Upstream weight license: Apache-2.0 (declared in the pinned `README.md` front matter; the snapshot carries no separate `LICENSE` file)
- Local snapshot: `weights/tapas-large-wtq/` with `dimer-base-manifest.json` (6 files, per-file bytes + SHA-256, `totalBytes` 1347256837); the Git repository does not vendor the checkpoint.
- Load-time check: `stage_missing_files()` then `verify_snapshot()` in `src/tapas_table_qa_pipeline/pipeline.py` — the first refuses a manifest naming another model or revision and fetches only missing manifest entries when `allow_download=True`; the second re-hashes every manifest entry and refuses on any mismatch.
- DIMER hosting: Apache-2.0 permits use, modification, distribution and commercial use subject to the license and notice requirements; DIMER may mirror the pinned checkpoint in its model store under the upstream license.
- Loader trust boundary: `transformers==4.57.6` built-in `TapasForQuestionAnswering` + `TapasTokenizer`, `trust_remote_code=False`, `local_files_only=True` from the snapshot directory; the checkpoint is the upstream default (`reset` position embeddings) branch, not the `no_reset` branch. Hub download is opt-in and pinned to the revision above.
