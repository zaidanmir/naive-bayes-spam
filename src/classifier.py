"""Multinomial Naive Bayes classifier from scratch.

The mathematical derivation lives in `notes/derivation.md`. In short:

    log P(c | x) ∝ log P(c) + Σ_w x_w · log P(w | c)

with Laplace-smoothed conditional probabilities

    P(w | c) = (n_{cw} + α) / (Σ_{w'} n_{cw'} + α |V|)

where n_{cw} is the count of word w in documents of class c, |V| is the
vocabulary size, and α > 0 is the smoothing constant.

"""
from __future__ import annotations

import numpy as np

from src.vocab import Vocabulary


class MultinomialNB:
    """Multinomial Naive Bayes with Laplace (additive) smoothing."""

    def __init__(self, vocab: Vocabulary, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive (use a small value, not 0)")
        self.vocab = vocab
        self.alpha = alpha
        self.classes_: np.ndarray | None = None
        self.log_prior_: np.ndarray | None = None
        self.log_likelihood_: np.ndarray | None = None

    def fit(self, texts, labels) -> "MultinomialNB":
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        n_classes = len(self.classes_)
        V = len(self.vocab)

        # Class priors: log of observed class frequencies.
        class_counts = np.array([(labels == c).sum() for c in self.classes_])
        self.log_prior_ = np.log(class_counts / class_counts.sum())

        # Per-class word totals: sum of count vectors over documents in each class.
        word_counts = np.zeros((n_classes, V), dtype=np.int64)
        for text, label in zip(texts, labels):
            class_idx = np.searchsorted(self.classes_, label)
            word_counts[class_idx] += self.vocab.encode(text)

        # Laplace-smoothed log-likelihood. Adding alpha to every count and dividing
        # by the smoothed row sum is equivalent to (n_{cw}+α) / (Σ n_{cw'} + α|V|),
        # since Σ_w (n_{cw}+α) = (Σ n_{cw}) + α|V|.
        smoothed = word_counts + self.alpha
        self.log_likelihood_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))

        return self

    def predict_log_proba(self, texts) -> np.ndarray:
        """Return un-normalised log P(c|x) per (text, class). Shape: (n_texts, n_classes).

        log P(c|x) ∝ log P(c) + Σ_w x_w · log P(w|c)

        The Σ_w sum is the dot product of the count vector with row c of
        log_likelihood_, vectorised across all texts as a single matmul of
        shape (n_texts, |V|) @ (|V|, n_classes).
        """
        if self.log_prior_ is None or self.log_likelihood_ is None:
            raise RuntimeError("predict_log_proba called before fit")
        encoded = np.stack([self.vocab.encode(t) for t in texts])
        return self.log_prior_ + encoded @ self.log_likelihood_.T

    def predict(self, texts) -> np.ndarray:
        """Return the predicted class label for each text."""
        return self.classes_[self.predict_log_proba(texts).argmax(axis=1)]


if __name__ == "__main__":
    from src.data import load_dataset, split_train_test

    df = load_dataset()
    train_df, _ = split_train_test(df)
    vocab = Vocabulary(min_count=2).fit(train_df["text"])

    clf = MultinomialNB(vocab=vocab, alpha=1.0)
    clf.fit(train_df["text"], train_df["is_spam"])

    print(f"Classes:           {clf.classes_}")
    print(f"Class proportions: {np.exp(clf.log_prior_)}")
    print(f"|V|:               {len(vocab):,}")
    print(f"log_likelihood_:   shape={clf.log_likelihood_.shape}")
    print()
    print("Most spam-indicative words (largest log P(w|spam) - log P(w|ham)):")
    diff = clf.log_likelihood_[1] - clf.log_likelihood_[0]
    for idx in np.argsort(diff)[-15:][::-1]:
        print(f"  {vocab.idx_to_word[idx]:<15}  {diff[idx]:+.3f}")

    print("\nSpot-check predictions on three handcrafted messages:")
    samples = [
        "Hey are you free for lunch tomorrow?",
        "WIN A FREE iPhone! Text WIN to 80082 to claim your prize now!!!",
        "Don't forget to grab milk on your way home",
    ]
    for text, pred in zip(samples, clf.predict(samples)):
        label = "spam" if pred == 1 else "ham"
        print(f"  [{label}]  {text!r}")
