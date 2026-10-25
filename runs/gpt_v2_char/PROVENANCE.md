# Provenance — runs/gpt_v2_char

Trained on Kaggle CPU (interactive session), 2026-10-25, commit `c9b0d9f` code
(zip `zerogpt.zip`), corpus `data/processed` at commit `040ae8d`. Session ran
3,000 steps in 1.67 h at 1,023 tokens/s, peak memory 4,385 MB, checkpoint
round-trip max logit difference 0.0.

The Kaggle session expired before its output files were downloaded, so this
directory was rebuilt locally the same day from the session's verbatim console
output:

- `results.json` — transcribed exactly from the training cell's printed JSON.
- `config.json` — reconstructed from the notebook's command line and results
  (all inputs deterministic).
- `tokenizer.json` — regenerated through the same code path
  (`CharacterTokenizer.from_text` on the train split; vocabulary matches the
  run's 100).
- `sample.txt` — the sample printed by the notebook's comparison cell,
  truncated at 2,000 characters by that cell's print statement (the full
  400-token sample is lost).

Not recoverable: `checkpoint.npz` (weights) and `train.jsonl` (per-step log).
Do not treat this run as reproducible-in-place from these files; a re-run
with seed 1337 on the same corpus and config should reproduce the reported
losses (V1 showed exact cross-machine determinism for this stack).
