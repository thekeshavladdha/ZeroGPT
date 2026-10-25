"""Load a saved GPT checkpoint and generate text from a custom prompt.

Usage:
    python scripts/sample_from_checkpoint.py --prompt "Sherlock Holmes" --temperature 0.8 --tokens 400
    python scripts/sample_from_checkpoint.py --prompt "The detective" --temperature 1.2 --tokens 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zerogpt.gpt import GPTModel
from zerogpt.tokenizer import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "gpt" / "checkpoint.npz")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "runs" / "gpt" / "tokenizer.json")
    parser.add_argument("--prompt", default="Sherlock Holmes")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    import json
    tokenizer = load_tokenizer(json.loads(args.tokenizer.read_text(encoding="utf-8")))
    model = GPTModel.load(args.checkpoint)

    prompt_tokens = tokenizer.encode(args.prompt).tolist()
    if not prompt_tokens:
        raise ValueError("The prompt contains characters outside the tokenizer vocabulary")

    rng = np.random.default_rng(args.seed)
    generated = list(prompt_tokens)
    for _ in range(args.tokens):
        context = np.asarray(generated[-model.config.context_length :], dtype=np.int64)[None, :]
        logits = model.forward(context).data[0, -1].astype(np.float64)
        logits = logits / args.temperature
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        generated.append(int(rng.choice(model.config.vocab_size, p=probabilities)))

    # Print safely: the decoded text may contain replacement characters from
        # malformed model output, which some terminal encodings cannot emit.
        # Write raw UTF-8 bytes to the underlying stdout buffer directly so
        # the terminal's text encoder is never in the critical path.
        text = tokenizer.decode(generated)
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            buffer.write(text.encode("utf-8", errors="replace") + b"\n")
            buffer.flush()
        else:
            sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()