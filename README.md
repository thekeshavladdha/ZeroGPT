# ZeroGPT

I wanted to understand how a GPT actually works, all the way down, so I built one from scratch in NumPy. No PyTorch, no transformer library. The tokenizer, the autograd engine, the attention, AdamW, the training loop, checkpointing, sampling: all of it is written by hand for this project, and the only dependency is NumPy. It trains on a regular CPU in minutes to a couple of hours.

Every number in this README comes from a saved run under `runs/`. If a result isn't in a results file somewhere, it isn't in here.

## The short version

The reason this project exists is the table below. I trained the same transformer twice on the same text, once with a character tokenizer and once with byte-pair encoding, under the same budget.

| Run | Tokenizer | Parameters | nats/token | nats/char |
| --- | --- | ---: | ---: | ---: |
| Bigram baseline | characters | - | 2.4184 | 2.4184 |
| GPT | characters | 1,842,916 | 1.4685 | 1.4685 |
| GPT | byte-pair, vocab 512 | 2,001,536 | 2.7792 | **1.3423** |

Byte-pair wins by 0.1263 nats per character.

Now look at the nats/token column. There, characters look better than BPE, 1.47 against 2.78. I believed that for a while. My first BPE run lost badly on per-token loss and I nearly wrote subwords off for this model size. Then it clicked: a BPE token covers about 2.07 characters on this corpus, so comparing nats per token across tokenizers with different token sizes is meaningless. Counted per character, that "losing" run had already won. Most of the work after that point exists because of that mistake, and because I wanted to re-run the fight at a scale where subwords should matter more.

There was an earlier round, too. The first version of this project was deliberately tiny: a character model with 115,544 parameters on the original single-book corpus, 6,000 steps, which reached 1.4863 nats and beat its bigram baseline (2.3788) by 0.89. That model and its run are still saved in `runs/gpt/`. The old single-book corpus lives in git history.

## What's hand-built

- A byte-level BPE tokenizer with a learned merge table, and a character tokenizer.
- A small reverse-mode autograd `Tensor` with exactly the ops this model needs: embedding lookup, linear, layer norm, causal attention, GELU, softmax cross-entropy.
- The decoder: pre-LayerNorm blocks, multi-head causal self-attention, GELU MLPs, learned position embeddings.
- AdamW with gradient clipping and warmup plus cosine decay.
- Deterministic batching, checkpoint save/load, and seeded sampling.

That's the whole stack. `requirements.txt` is one line: `numpy`.

## Data

V2 trains on seven public-domain Project Gutenberg books: four Sherlock Holmes books (Adventures, Memoirs, The Hound of the Baskervilles, The Return), one Holmes short story (Wisteria Lodge; ebook 2343 turned out to contain just that story rather than all of His Last Bow), plus Tom Sawyer and Huckleberry Finn, because I wanted a very different voice in the mix. Sources, licenses, and the cleaning rules are in [`data/README.md`](data/README.md).

```bash
python data/prepare.py
```

This downloads each book once (cached in `data/raw/`), strips the Gutenberg boilerplate and illustration captions, and reflows paragraphs. Cleaned, the corpus is 3,029,648 characters with a vocabulary of 100, split 90/10 in document order: 2,726,749 training characters, 302,899 validation. The validation split is the tail of the corpus, so no text is shuffled across the boundary.

## Running it

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The bigram baseline takes seconds:

```bash
python scripts/train_bigram.py --output-dir runs/bigram
```

The original small model (its saved run was trained on the old single-book corpus):

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

The two V2 runs from the table above:

```bash
python scripts/train_gpt.py --steps 3000 --batch-size 16 --eval-batch-size 64 --eval-interval 100 --warmup-steps 200 --learning-rate 3e-3 --num-layers 4 --num-heads 6 --model-width 192 --mlp-width 768 --context-length 128 --sample-tokens 400 --tokenizer-type character --output-dir runs/gpt_v2_char

python scripts/train_gpt.py --steps 3000 --batch-size 16 --eval-batch-size 64 --eval-interval 100 --warmup-steps 200 --learning-rate 3e-3 --num-layers 4 --num-heads 6 --model-width 192 --mlp-width 768 --context-length 128 --sample-tokens 400 --tokenizer-type byte_pair --vocab-limit 512 --output-dir runs/gpt_v2_bpe
```

Each run writes `config.json`, `train.jsonl`, `results.json`, `tokenizer.json`, `checkpoint.npz`, and `sample.txt` to its output directory.

## How the comparison was run

Both models share everything: 4 layers, width 192, MLP width 768, 6 heads, context 128, batch 16, 3,000 steps, learning rate 3e-3 with 200 warmup steps and cosine decay, seed 1337, same corpus. Tokenizers are fitted on the training split only; an earlier version of my script fitted BPE merges on train plus validation, which was a leak, so that's fixed and covered by tests now. I picked the BPE vocabulary (512, over 1024 and 2048) and the learning rate with short 500-step pilots on my laptop, then did the real runs on Kaggle's CPU, about two hours each.

BPE gets a second advantage worth naming. At 2.07 characters per token, the same 3,000 steps push roughly 12.7M characters through the BPE model against 6.1M for the character model. That isn't cheating, that's the point of compression.

Two footnotes I want on the record. First, the `beats_bigram` flag stored in the BPE run's `results.json` reads false, because the script compares nats per token against a character bigram. Same trap as the main table, just in flag form. Second, I lost the character run's files: the Kaggle session expired before I downloaded anything, and its checkpoint and step log are gone. [`runs/gpt_v2_char/`](runs/gpt_v2_char) holds the numbers transcribed from that session's printed output, and the `PROVENANCE.md` inside says exactly what survived and what didn't. The numbers are real, the file set isn't complete, and I learned the download-before-closing-the-tab lesson the annoying way. The BPE run in [`runs/gpt_v2_bpe/`](runs/gpt_v2_bpe) is complete, and I verified its tokenizer reproduces the committed code's tokenization exactly before trusting it. An earlier BPE run that couldn't pass that check went in the bin.

## Samples

Both samples use the prompt `Sherlock Holmes`, seed 1337, temperature 0.8.

The V1 character model (full text in [`runs/gpt/sample.txt`](runs/gpt/sample.txt)):

```text
Sherlock Holmes. "You have seen at that they father all."

“Amerious a sharmida in in conving the room.

“In go you will go one it was nevilled over a good into myself, I it still ways the double upon our instant. There?”

“But it is may be the trace, and in of year end coatual hair keen and of my peave that send to remarked uncle, the was nowing dropped of the storibling hydred gestable the case that first t
```

The V2 BPE model (from [`runs/gpt_v2_bpe/sample.txt`](runs/gpt_v2_bpe/sample.txt)):

```text
Sherlock Holmes’s faces are guess I am on a long for cases, as you makes a sortnal on the rideg of
tichen-conclusion. Yes, I don’t see a little, I should have probably going to me, for I have made
thrown from it.”

“Ah, Mortimer,” said Holmes, “they has since you get me ”
```

The character model spells fine and mostly finds real words, but the phrases wander. The BPE model holds phrases together, gets the dialogue punctuation right, and pulls actual names from the books; Mortimer walks in from The Hound of the Baskervilles. Both are still visibly broken, which feels about right for this parameter count and a two-hour CPU budget.

## On Kaggle

[`notebooks/zerogpt_kaggle.ipynb`](notebooks/zerogpt_kaggle.ipynb) runs the whole pipeline on Kaggle's CPU: it unpacks the repo, preps the corpus, trains the baseline and the V2 models, prints the comparison, and zips `runs/` for download. About two hours per training run, and one run per session is comfortable. The T4 doesn't help; this is NumPy on CPU by design.

Two practical things I learned up there. Interactive working files die with the session, so download the outputs (or hit Save Version) before closing the tab. And verify what you're actually running: the notebook has a small cell that checks the corpus and the exact BPE tokenization before you burn two hours on training.

## Tests and rough edges

The suite covers autograd gradients and causal masking, both tokenizers, batching, the bigram model, transformer causality, AdamW, overfitting on a fixed batch, sampling, and checkpoint round-trips:

```bash
python -m unittest discover -s tests
```

Rough edges I know about:

- The character V2 run has no checkpoint or step log, as described above.
- No GPU backend and no KV cache. Sampling works, it's just honest about being slow.
- Sample quality swings with the seed and temperature, so don't read much into any single sample.
- This is a demonstration. It's not a language product and I don't pretend otherwise.
