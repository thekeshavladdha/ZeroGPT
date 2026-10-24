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

The supplied `dataset.txt` is Project Gutenberg eBook **Adventures of Sherlock Holmes**, ebook 48320, by Arthur Conan Doyle. The source, license information, cleaning rules, and sharing considerations are recorded in [`data/README.md`](data/README.md).

Prepare the corpus from the repository root:

```bash
python data/prepare.py
```

The current prepared corpus contains:

| Artifact | Value |
| --- | ---: |
| Cleaned characters | 562,585 |
| Vocabulary size | 88 |
| Training characters | 506,455 |
| Validation characters | 56,130 |
| Actual training fraction | 0.900228 |

The split is document-ordered and uses the paragraph boundary nearest the requested 90% boundary. The prepared files are written to `data/processed/`.

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
  --steps 1200 \
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
| Tiny GPT | 1.5602985950 | 5,806 tokens/s | 510.32 MB |

The GPT improves the saved validation loss over the bigram baseline by **0.8184646132 nats**. The checkpoint reload check recorded a maximum absolute logit difference of **0.0**.

The throughput and memory numbers above are from the local Linux development runner. They are not a Kaggle hardware measurement. The Kaggle notebook is provided for a reproducible CPU run in that environment.

## Generated sample

The fixed showcase prompt is `Sherlock Holmes`, with seed `1337` and temperature `0.8`. The full generated output is saved in [`runs/gpt/sample.txt`](runs/gpt/sample.txt):

```text
Sherlock Holmes. “You have the gavellentiess, Mr.” easked craid, to he that the kind seems, reson the past rain with some of his proloyed which mury can the crime, but have any to way, sat he concentratler surb it was not oncent some of his loveer fierect mans enough folow for and his color and of his happed as of man. Stome homn and it from to to comportable out his that to husborne room for and the influffi
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

- The model is character-level and has only a 32-character context.
- The 32,216-parameter model is a demonstration, not a language product.
- Sampling quality depends strongly on the fixed training budget, seed, and temperature.
- There is no GPU backend, KV cache, BPE tokenizer, or scaling study.
- The generated sample still requires the planned human review gate.
- The public release should wait for owner approval after the Kaggle run and sample review.
