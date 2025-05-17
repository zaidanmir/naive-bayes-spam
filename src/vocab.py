"""Tokenisation and vocabulary construction for the spam classifier.

The classifier consumes documents as count vectors over a fixed vocabulary,
so this module is responsible for two things:

1. `tokenize(text)` — turn raw SMS text into a sequence of lowercase tokens
2. `Vocabulary` — fit a word ↔ index mapping on a training corpus and expose
   `encode(text)` to turn a single message into a count vector
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np

# Match runs of letters and apostrophes — drops digits, punctuation, currency symbols.
_TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    """Lowercase, then extract alphabetic tokens (apostrophes preserved)."""
    return _TOKEN_RE.findall(text.lower())


class Vocabulary:
    """Word ↔ index mapping built from a training corpus."""

    def __init__(self, min_count: int = 1) -> None:
        self.min_count = min_count
        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: list[str] = []

    def fit(self, texts: Iterable[str]) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        # Drop tokens below min_count to cut noise from typos and one-off tokens
        # that would otherwise inflate the vocabulary without adding signal.
        kept = sorted(w for w, c in counter.items() if c >= self.min_count)
        self.idx_to_word = kept
        self.word_to_idx = {w: i for i, w in enumerate(kept)}
        return self

    def __len__(self) -> int:
        return len(self.idx_to_word)

    def encode(self, text: str) -> np.ndarray:
        """Return a 1-D count vector of shape (|V|,) for a single document."""
        counts = np.zeros(len(self), dtype=np.int32)
        for tok in tokenize(text):
            idx = self.word_to_idx.get(tok)
            if idx is not None:
                counts[idx] += 1
        return counts


if __name__ == "__main__":
    from src.data import load_dataset, split_train_test

    df = load_dataset()
    train_df, _ = split_train_test(df)

    for min_count in (1, 2, 5):
        vocab = Vocabulary(min_count=min_count).fit(train_df["text"])
        print(f"min_count={min_count}: |V| = {len(vocab):,}")

    vocab = Vocabulary(min_count=2).fit(train_df["text"])
    sample = train_df["text"].iloc[0]
    counts = vocab.encode(sample)
    print(f"\nFirst training message:\n  {sample!r}")
    print(f"  non-zero tokens: {int((counts > 0).sum())}, sum: {int(counts.sum())}")
