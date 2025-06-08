"""scikit-learn MultinomialNB baseline for sanity-checking the from-scratch implementation.

If our implementation is correct (and the tokenisation / vocabulary policy
is comparable), the two should agree to within rounding error.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB as SKMultinomialNB

from src.classifier import MultinomialNB as OursMultinomialNB
from src.data import load_dataset, split_train_test
from src.evaluate import accuracy, precision_recall_f1
from src.vocab import Vocabulary, tokenize

RESULTS_PATH = Path(__file__).resolve().parent / "results.md"


def evaluate_ours(train_df, test_df, *, min_count=2, alpha=1.0):
    vocab = Vocabulary(min_count=min_count).fit(train_df["text"])
    clf = OursMultinomialNB(vocab=vocab, alpha=alpha)
    clf.fit(train_df["text"], train_df["is_spam"])
    y_true = test_df["is_spam"].to_numpy()
    y_pred = clf.predict(test_df["text"])
    p, r, f1 = precision_recall_f1(y_true, y_pred, positive_class=1)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
        "vocab_size": len(vocab),
    }


def evaluate_sklearn(train_df, test_df, *, min_df=2, alpha=1.0):
    # Use our tokenizer for an apples-to-apples comparison; the only thing
    # being benchmarked is the MultinomialNB implementation, not preprocessing.
    vectorizer = CountVectorizer(tokenizer=tokenize, min_df=min_df, lowercase=False)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    clf = SKMultinomialNB(alpha=alpha)
    clf.fit(X_train, train_df["is_spam"])
    y_true = test_df["is_spam"].to_numpy()
    y_pred = clf.predict(X_test)
    p, r, f1 = precision_recall_f1(y_true, y_pred, positive_class=1)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
        "vocab_size": len(vectorizer.vocabulary_),
    }


def write_results(ours, theirs) -> None:
    rows = [
        ("Implementation", "Accuracy", "Precision (spam)", "Recall (spam)", "F1 (spam)", "|V|"),
        (
            "From-scratch (this repo)",
            f"{ours['accuracy']:.4f}",
            f"{ours['precision']:.4f}",
            f"{ours['recall']:.4f}",
            f"{ours['f1']:.4f}",
            f"{ours['vocab_size']:,}",
        ),
        (
            "scikit-learn `MultinomialNB`",
            f"{theirs['accuracy']:.4f}",
            f"{theirs['precision']:.4f}",
            f"{theirs['recall']:.4f}",
            f"{theirs['f1']:.4f}",
            f"{theirs['vocab_size']:,}",
        ),
    ]
    md = ["# Benchmark — from-scratch vs scikit-learn", ""]
    md.append("Both runs use the same data, the same tokenizer (`src/vocab.tokenize`),")
    md.append("the same vocabulary cutoff (`min_count=2`), and the same Laplace")
    md.append("smoothing (`alpha=1.0`). Only the classifier implementation differs.")
    md.append("")
    md.append("| " + " | ".join(rows[0]) + " |")
    md.append("|" + "|".join(["---"] * len(rows[0])) + "|")
    for row in rows[1:]:
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    delta = abs(ours["accuracy"] - theirs["accuracy"])
    md.append(f"Accuracy delta: **{delta:.4f}**.")
    md.append("")
    md.append("Reproduce: `python -m bench.sklearn_baseline`.")
    RESULTS_PATH.write_text("\n".join(md))


def main() -> None:
    df = load_dataset()
    train_df, test_df = split_train_test(df)

    ours = evaluate_ours(train_df, test_df)
    theirs = evaluate_sklearn(train_df, test_df)

    print(f"{'metric':<22}{'from-scratch':>15}{'sklearn':>12}{'delta':>10}")
    for k in ("accuracy", "precision", "recall", "f1"):
        d = ours[k] - theirs[k]
        print(f"{k:<22}{ours[k]:>15.4f}{theirs[k]:>12.4f}{d:>+10.4f}")
    print(f"\n|V| ours: {ours['vocab_size']:,}    |V| sklearn: {theirs['vocab_size']:,}")

    write_results(ours, theirs)
    print(f"\nResults written to {RESULTS_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
