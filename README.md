# ZeroGPT

ZeroGPT is a small, reproducible decoder-only GPT trained from scratch on the supplied **Adventures of Sherlock Holmes** text. It is a model-building demonstration: the tokenizer, autograd operations, transformer, optimizer, training loop, checkpointing, and sampler are implemented with NumPy and the Python standard library.

**Maintainer:** project owner

## Scope

The project follows the plan in [`roadmap.md`](roadmap.md). V1 keeps the model intentionally small and reviewable:

- Character-level tokenizer with a vocabulary derived from the cleaned corpus.
- Count-based character bigram baseline.
- Decoder-only transformer with causal multi-head self-attention.
- Custom NumPy reverse-mode autograd.
- AdamW with gradient clipping.
- Checkpoint save/load and seeded text sampling.

The implementation uses only the allowed runtime dependency in [`requirements.txt`](requirements.txt): NumPy. Pandas, PyTorch, SciPy, CuPy, and other numerical frameworks are not required.

## Data

V2 trains on seven public-domain Project Gutenberg ebooks: the Sherlock Holmes canon by Arthur Conan Doyle (1661, 834, 2852, 108, 2343) plus Mark Twain's *Tom Sawyer* (74) and *Huckleberry Finn* (76). Sources, license notes, and cleaning rules are recorded in [`data/README.md`](data/README.md).

Prepare the corpus from the repository root:

```bash
python data/prepare.py
```

The current prepared corpus contains:

| Artifact | Value |
| --- | ---: |
| Cleaned characters | 3,029,648 |
| Vocabulary size | 100 |
| Training characters | 2,726,749 |
| Validation characters | 302,899 |
| Actual training fraction | 0.900022 |

The split is document-ordered and uses the paragraph boundary nearest the requested 90% boundary. V1's single-book corpus (562,585 characters, vocabulary 88) remains in git history.

## Run locally

Install the declared dependency and run the tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the baseline:

```bash
python scripts/train_bigram.py --output-dir runs/bigram
```

Run the tiny GPT with the saved showcase configuration:

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

The training command writes `config.json`, `train.jsonl`, `results.json`, `tokenizer.json`, `checkpoint.npz`, and `sample.txt` to its output directory. The GPT runner reads the baseline result from `runs/bigram/results.json` when it records the comparison.

## Model

The saved showcase model uses this configuration from [`runs/gpt/config.json`](runs/gpt/config.json):

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

Each decoder block applies LayerNorm, causal multi-head self-attention, a residual connection, a GELU MLP, and another residual connection. Learned token and position embeddings feed the stack, and a final LayerNorm plus linear head produces character logits.

## Saved results

All numbers below come from the saved JSON result files, not from an estimate in this document.

| Run | Validation loss (nats) | Training throughput | Peak memory |
| --- | ---: | ---: | ---: |
| Character bigram | 2.3787632082 | 11,718,388 tokens/s | 36.71 MB |
| Tiny GPT | 1.4862852312 | 5,202 tokens/s | 520.58 MB |

The GPT improves the saved validation loss over the bigram baseline by **0.8924779770 nats**. The checkpoint reload check recorded a maximum absolute logit difference of **0.0**.

The throughput and memory numbers above are measured on the Kaggle CPU runner via `notebooks/zerogpt_kaggle.ipynb`. The T4 does not accelerate this NumPy-only implementation.

## V2: BPE versus character tokenization

V2 asks which tokenizer wins when the model is large enough for subwords to pay off. All cross-tokenizer claims use validation **loss per character**: BPE compresses text ~2.07×, so per-token losses are not comparable across tokenizers (V1's BPE run "losing" at 3.02 nats/token while already winning at 1.41 nats/char was exactly this artifact).

Both V2 models share the architecture (4 decoder layers, width 192, MLP 768, 6 heads, context 128, batch 16) and an identical budget: 3,000 steps, learning rate 3e-3 with warmup and cosine decay, seed 1337, on the 3.03M-character corpus. Tokenizers are fitted on the training split only; 500-step local pilots selected BPE vocabulary 512 and learning rate 3e-3 for both sides.

| Run | Tokenizer | Parameters | Val loss (nats/token) | **Val loss (nats/char)** |
| --- | --- | ---: | ---: | ---: |
| Bigram baseline | character | — | 2.4184 | 2.4184 |
| GPT V2 character | character | 1,842,916 | 1.4685 | 1.4685 |
| GPT V2 BPE-512 | byte-pair (2.07 chars/token) | 2,001,536 | 2.7792 | **1.3423** |

**BPE wins by 0.1263 nats/char (8.6% lower)** at the same step count, and every BPE token carried ~2.07 characters, so the BPE run processed ~12.7M characters of context against the character run's ~6.1M within the same budget. Both runs beat the bigram baseline on nats/char. (The `beats_bigram` flag inside the BPE run's `results.json` compares nats/token against the character bigram — the same cross-metric illusion; the nats/char column is the honest comparison.)

Artifacts: [`runs/gpt_v2_bpe/`](runs/gpt_v2_bpe) holds the full Kaggle run (checkpoint verified to reload with 0.0 logit difference); [`runs/gpt_v2_char/`](runs/gpt_v2_char) results were transcribed from that session's printed output after the session expired before download — see its `PROVENANCE.md`; the character checkpoint and step log are not recoverable. The BPE sample below is from `runs/gpt_v2_bpe/sample.txt` (prompt "Sherlock Holmes", seed 1337, temperature 0.8):

```text
Sherlock Holmes’s faces are guess I am on a long for cases, as you makes a sortnal on the rideg of
tichen-conclusion. Yes, I don’t see a little, I should have probably going to me, for I have made
thrown from it.”

“Ah, Mortimer,” said Holmes, “they has since you get me ”
```

Compared with the character-level sample above, the BPE model produces longer coherent phrases, correct dialogue formatting, and corpus names (Mortimer), while remaining visibly imperfect at this scale.

## Generated sample

The fixed showcase prompt is `Sherlock Holmes`, with seed `1337` and temperature `0.8`. The full generated output is saved in [`runs/gpt/sample.txt`](runs/gpt/sample.txt):

```text
Sherlock Holmes. "You have seen at that they father all."

“Amerious a sharmida in in conving the room.

“In go you will go one it was nevilled over a good into myself, I it still ways the double upon our instant. There?”

“But it is may the trace, and in of year end coatual hair keen and of my peave that send to remarked uncle, the was nowing dropped of the storibling hydred gestable the case that first t
```

This sample is evidence for review, not a quality guarantee. It contains learned character and phrase patterns but remains visibly imperfect at this model size and training budget.

## Kaggle runner

Open [`notebooks/zerogpt_kaggle.ipynb`](notebooks/zerogpt_kaggle.ipynb) in Kaggle after making the repository available as a Kaggle dataset or working-directory upload. The notebook:

1. Locates the repository root.
2. Installs the declared requirements.
3. Re-prepares the corpus.
4. Runs the bigram baseline.
5. Runs the tiny GPT entry point.
6. Prints the saved results and packages the `runs/` directory as an artifact.

NumPy operations run on the Kaggle CPU; the T4 does not accelerate this implementation.

## Tests and limitations

The project uses focused `unittest` checks for:

- Autograd gradients and causal attention.
- Tokenizer and deterministic batching.
- Bigram fitting, evaluation, and checkpoint reload.
- Transformer causality, AdamW updates, fixed-batch overfitting, sampling, and GPT checkpoint reload.

Current limitations:

- The models are demonstrations at 115K (V1) and ~1.8-2.0M (V2) parameters with 64- and 128-token contexts — not language products.
- Sampling quality depends strongly on the fixed training budget, seed, and temperature; both saved samples remain visibly imperfect.
- There is no GPU backend or KV cache; all training ran on CPU (Kaggle and local), with the T4 providing no acceleration.
- The character V2 run's checkpoint and step log were lost with an expired Kaggle session; its saved numbers are transcribed session output (see `runs/gpt_v2_char/PROVENANCE.md`).
- The public release should wait for owner approval of the final packet.
