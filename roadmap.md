# ZeroGPT Roadmap

> V1 goal: build and showcase a small decoder-only GPT trained on the provided Sherlock Holmes text. Keep the project narrow, reproducible, and understandable.

## What success looks like

A reviewer can open the repository, understand the model, reproduce a small training run, inspect the measured result, reload the checkpoint, and see generated text.

V1 is complete when:

1. Gutenberg framing and license boilerplate are removed from the training corpus; the story text and story headings are retained. The source and license are recorded in data/README.md.
2. The project has a character tokenizer and a simple character-level bigram baseline.
3. A small GPT with our own model and backward-pass code trains on the prepared text.
4. Its validation loss beats the bigram baseline, its checkpoint reloads to identical logits, and it generates readable Sherlock-style samples.
5. The repository contains the config, seed, logs, benchmark results, sample output, and a minimal Kaggle notebook so the run can be reviewed and reproduced.
6. The README explains the architecture, how to run it, the measured results, and the model's limitations. Every reported number comes from a saved run result.

The model is a demonstration, not a language product. Do not make quality claims beyond the measured loss and the samples.

## Agreed constraints

- Implement the tokenizer, autograd, transformer, optimizer, training loop, checkpointing, and sampling ourselves.
- Use NumPy arrays and vectorized numerical operations for speed. Keep the custom autograd and model code ours; do not use a framework that supplies them.
- The only permitted third-party libraries are NumPy and pandas. Pandas is optional and should only be used if it materially helps; plain text preparation can use the standard library.
- Do not add PyTorch, SciPy, CuPy, or other packages.
- Kaggle's T4 will not accelerate this NumPy/pandas-only implementation. Train on the Kaggle CPU and measure actual throughput and peak memory. If performance is unacceptable, stop and revisit the dependency decision before adding a GPU backend.
- Correctness comes before optimization. Start with vectorized NumPy operations, profile the working training step, then optimize only measured bottlenecks. Save tokens per second, peak memory, config, and seed with each run.
- There is no calendar deadline. This is a side project; use reviewable milestones.
- Use unittest and keep tests focused on the operations and behaviors the model depends on.

## V1 model

Start with the existing tiny preset: 2 decoder layers, 4 attention heads, model width 32, and context length 32. Derive vocabulary from the cleaned corpus. Treat this as a starting point; change it only when a measured result gives a reason.

Use a character tokenizer, learned position embeddings, causal multi-head self-attention, LayerNorm, and a small GELU MLP. Keep the implementation to the shapes and operations this model needs. Do not build a general-purpose tensor framework.

The NumPy-backed Tensor type owns its gradient accumulation and backward pass. Required operations include embedding lookup, linear layers, normalization, causal attention, activation, softmax cross-entropy, and the reductions needed for training. AdamW, clipping, sampling, and checkpoint save/load are implemented in the project.

## Data and evaluation

- Clean the Project Gutenberg wrapper and license text before splitting. Keep the stories and their headings.
- Split in document order: first 90% for training, last 10% for validation. Never shuffle neighboring text into both sets.
- Train the bigram baseline and GPT on the same split. Compare validation cross-entropy in nats.
- Save each run's config, seed, step log, validation result, tokens per second, peak memory, checkpoint, and generated sample.
- Use a fixed prompt and seed for the showcase sample. Human review of the sample is a separate gate from the numeric loss comparison.

## Review and ownership

The coding agent prepares code, tests, configs, experiment artifacts, and documentation. The project owner reviews work in milestone packets rather than writing code.

Each packet contains changed files, checks run, result files, a short explanation of the design, and any decision needed. Pause at each gate for review before moving to the next gate. The final public-facing release waits for the owner's approval.

List the project owner as the sole project owner/maintainer. The README does not need a separate AI-assistance section. Keep all descriptions accurate; do not claim personal hand-authorship of specific code unless that is true. Follow any disclosure rule imposed by a publication venue.

## Phases

### Phase 0: Prepare the corpus and project

- Record the dataset source and license.
- Prepare cleaned train and validation text and report character count and vocabulary size.
- Add only the minimal package layout, dependency file, and allowlist check needed for NumPy and optional pandas.
- Gate: inspect the cleaned text, split boundary, source note, and preparation summary.

### Phase 1: Build and verify the NumPy-backed engine

- Implement only the Tensor/autograd operations the small GPT needs.
- Add finite-difference gradient checks for each differentiable operation, plus a causality check for attention.
- Gate: checks pass, a tiny fixed batch can be overfit, and the owner receives an explanation of the backward path.

### Phase 2: Establish the CPU baseline

- Implement the character tokenizer, count-based bigram model, deterministic batching, evaluation, and a small training loop.
- Measure CPU tokens per second and peak memory on Kaggle. Use this measurement to confirm the tiny GPT run budget.
- Gate: bigram results and throughput are saved to files; the run can be repeated with the same config and seed.

### Phase 3: Train the tiny GPT

- Implement the decoder-only model, AdamW, checkpoint save/load, and sampling.
- Train against the prepared corpus and compare validation loss with the bigram baseline.
- Verify causal masking, checkpoint round-trip (identical logits), and overfit on one fixed batch.
- Gate: baseline comparison, logs, checkpoint, generated sample, and CPU performance result are saved; the sample is reviewed by the owner.

### Phase 4: Package the showcase

- Provide a minimal Kaggle notebook that runs the repository's training entry point and saves its outputs.
- Write a concise README with architecture, setup and run steps, measured results, generated text, source/license, and limitations.
- Keep the public-ready v1 to the repository, Kaggle runner, README, and actual results. Publish only after the owner approves the final packet.

## Out of scope for V1

Do not build BPE, RoPE, RMSNorm, SwiGLU, GQA, KV cache, a browser inference demo, a large ablation/scaling study, native acceleration, a blog, or a video. Consider an addition only after the tiny GPT passes all v1 gates and there is a concrete reason to add it.

## V2 — tokenizer study (2026-10-25)

With the V1 gates passed, the next question was the one V1 had parked: does byte-pair tokenization pay off at a larger scale?

What changed: the corpus grew to seven public-domain books (3,029,648 characters), the training script gained architecture flags, tokenizers are now fitted on the training split only (the old code fit on train plus validation, a small leak), and every run records validation loss per character. That last one matters because V1's BPE comparison at 3.02 nats/token was a measurement mistake; per character, that run had already won (1.41 vs 1.49).

Both tokenizers trained the same model (4 layers, width 192, MLP 768, 6 heads, context 128, batch 16; about 1.84M parameters character, 2.00M BPE-512) for 3,000 steps at learning rate 3e-3 with warmup and cosine decay, seed 1337. Short 500-step local runs picked BPE vocabulary 512 and learning rate 3e-3 for both sides.

Gates agreed before running anything: BPE beats character by at least 0.05 nats/char, both beat the bigram baseline, the owner reviews the BPE sample, and a character context-256 control run happens only if the margin lands under 0.1.

Result: BPE 1.3423 vs character 1.4685 nats/char, margin 0.1263, so no control run was needed. Both beat the 2.4184 bigram. One earlier Kaggle BPE run was discarded because its tokenizer could not be reproduced from the committed code; the notebook now checks the corpus and the exact tokenization before training starts. Full artifacts in `runs/gpt_v2_bpe/`; `runs/gpt_v2_char/` carries the transcribed numbers, with a `PROVENANCE.md` explaining what survived a lost session.