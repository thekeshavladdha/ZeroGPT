# ZeroGPT

ZeroGPT is a small decoder-only GPT built from scratch in NumPy. No PyTorch, no transformer library: the tokenizer, the autograd engine, the transformer, AdamW, the training loop, checkpointing, and the sampler are all written by hand for this project, using only NumPy and the standard library. It trains on a plain CPU, and every number in this README comes from a saved run under `runs/`.

## Scope

V1 was kept small on purpose: a character tokenizer, a count-based bigram baseline, and a ~115K-parameter transformer on the supplied Sherlock Holmes text. V2 grew the corpus and the model to answer one question: does byte-pair tokenization beat character-level tokenization once the model is big enough for subwords to matter? Short answer: yes, and V1's BPE "failure" was a measurement mistake, not a real one.

The only runtime dependency is NumPy ([`requirements.txt`](requirements.txt)). No pandas, PyTorch, SciPy, or CuPy anywhere.

## Data

V2 trains on seven public-domain Project Gutenberg books: the Sherlock Holmes canon (ebooks 1661, 834, 2852, 108, 2343) plus *Tom Sawyer* and *Huckleberry Finn* (74, 76). Sources, license notes, and the cleaning rules are in [`data/README.md`](data/README.md).

```bash
python data/prepare.py
```

The prepared corpus:

| Artifact | Value |
| --- | ---: |
| Cleaned characters | 3,029,648 |
| Vocabulary size | 100 |
| Training characters | 2,726,749 |
| Validation characters | 302,899 |
| Actual training fraction | 0.900022 |

The split stays document-ordered, cutting at the paragraph boundary nearest 90%. V1's single-book corpus (562,585 characters, 88 unique characters) is still in git history.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Train the baseline:

```bash
python scripts/train_bigram.py --output-dir runs/bigram
```

Reproduce the V1 showcase run (character tokenizer, 6,000 steps):

```bash
python scripts/train_gpt.py \
  --steps 6000 \
  --tokenizer-type character \
  --batch-size 16 \
  --eval-batch-size 64 \
  --eval-interval 100 \
  --sample-tokens 400 \
  --output-dir runs/gpt
```

Every training run writes `config.json`, `train.jsonl`, `results.json`, `tokenizer.json`, `checkpoint.npz`, and `sample.txt` to its output directory, and records its comparison against `runs/bigram/results.json`.

## Model

The V1 showcase model ([`runs/gpt/config.json`](runs/gpt/config.json)):

| Setting | Value |
| --- | ---: |
| Decoder layers | 2 |
| Attention heads | 8 |
| Model width | 64 |
| MLP width | 256 |
| Context length | 64 |
| Vocabulary size | 88 |
| Parameters | 115,544 |
| Seed | 1337 |

Each block is LayerNorm, causal multi-head attention, a residual connection, a GELU MLP, and another residual. Learned token and position embeddings feed the stack; a final LayerNorm and linear head produce the logits. The V2 model is the same design scaled to 4 layers, width 192, MLP 768, 6 heads, context 128.

## Saved results (V1)

| Run | Validation loss (nats) | Throughput | Peak memory |
| --- | ---: | ---: | ---: |
| Character bigram | 2.3787632082 | 11,718,388 tokens/s | 36.71 MB |
| Tiny GPT | 1.4862852312 | 5,202 tokens/s | 520.58 MB |

The GPT beats the bigram by 0.8924779770 nats and its checkpoint reloads with a maximum logit difference of 0.0. Throughput was measured on the Kaggle CPU runner; the T4 does nothing for this code.

## V2: byte-pair versus character

The first BPE attempt looked like a loss: 3.02 nats per token against 1.49 for characters. That comparison was wrong. A BPE token covers about 2.07 characters on this corpus, so per-token losses don't mean the same thing for the two tokenizers. Counted per character, that same BPE run had already won (1.41 vs 1.49). V2 was built to test the question properly at a size where subwords should matter.

Both models share the same architecture and budget:

- 4 layers, width 192, MLP width 768, 6 heads, context 128, batch 16
- 3,000 steps, learning rate 3e-3 (200-step warmup, cosine decay), seed 1337
- the 3.03M-character corpus with its 90/10 split
- tokenizers fitted on the training split only
- BPE vocabulary (512) and learning rate picked by short 500-step local runs

| Run | Tokenizer | Parameters | nats/token | nats/char |
| --- | --- | ---: | ---: | ---: |
| Bigram | character | - | 2.4184 | 2.4184 |
| GPT | character | 1,842,916 | 1.4685 | 1.4685 |
| GPT | byte-pair 512 | 2,001,536 | 2.7792 | **1.3423** |

BPE wins by 0.1263 nats per character, about 8.6% lower, at the same step count. It also got more out of each step: with 2.07 characters per token, the BPE run pushed roughly 12.7M characters through the model against the character run's 6.1M. (The `beats_bigram` flag stored in the BPE run's `results.json` compares nats per token against a character bigram and reads false for the same reason. Quote the per-character column.)

For the record: the character run's files were rebuilt from the session's printed output after a Kaggle session expired before download, so [`runs/gpt_v2_char/`](runs/gpt_v2_char) has no checkpoint or step log (its `PROVENANCE.md` explains exactly what survived). [`runs/gpt_v2_bpe/`](runs/gpt_v2_bpe) is the complete run, and its tokenizer was checked to reproduce the committed code's tokenization exactly.

## Samples

The V1 showcase sample (prompt `Sherlock Holmes`, seed 1337, temperature 0.8, full text in [`runs/gpt/sample.txt`](runs/gpt/sample.txt)):

```text
Sherlock Holmes. "You have seen at that they father all."

“Amerious a sharmida in in conving the room.

“In go you will go one it was nevilled over a good into myself, I it still ways the double upon our instant. There?”

“But it is may be the trace, and in of year end coatual hair keen and of my peave that send to remarked uncle, the was nowing dropped of the storibling hydred gestable the case that first t
```

And the V2 BPE sample (from [`runs/gpt_v2_bpe/sample.txt`](runs/gpt_v2_bpe/sample.txt), same prompt, seed, and temperature):

```text
Sherlock Holmes’s faces are guess I am on a long for cases, as you makes a sortnal on the rideg of
tichen-conclusion. Yes, I don’t see a little, I should have probably going to me, for I have made
thrown from it.”

“Ah, Mortimer,” said Holmes, “they has since you get me ”
```

The BPE text holds phrases together better, gets the dialogue formatting right, and pulls real names from the corpus (Mortimer). Still visibly broken, which is about right for 2M parameters and a two-hour CPU run.

## Kaggle runner

Open [`notebooks/zerogpt_kaggle.ipynb`](notebooks/zerogpt_kaggle.ipynb) on Kaggle with the repository attached as a dataset or uploaded to the working directory. The notebook extracts the repo, installs requirements, prepares the corpus, runs the bigram baseline, trains the V2 models (one run per session is comfortable, about two hours each), prints the comparison, and zips `runs/` for download. Everything runs on the Kaggle CPU; the T4 does not accelerate this implementation.

## Tests and limitations

Focused `unittest` checks cover autograd gradients and causal attention, tokenizers and batching, bigram fitting and reload, transformer causality, AdamW, fixed-batch overfitting, sampling, and checkpoint reload.

Known limitations:

- These are demonstrations (115K parameters for V1, about 1.8-2.0M for V2), not language products.
- Sample quality depends heavily on the training budget, seed, and temperature.
- CPU only. No GPU backend and no KV cache.
- The character V2 run has no checkpoint or step log (see `runs/gpt_v2_char/PROVENANCE.md`).
- Public release waits on the owner's final approval.
