"""End-to-end training script: data -> vocab -> classifier -> metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.classifier import MultinomialNB
from src.data import load_dataset, split_train_test
from src.evaluate import accuracy, classification_report, confusion_matrix, precision_recall_f1
from src.vocab import Vocabulary

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing constant")
    parser.add_argument("--min-count", type=int, default=2, help="Vocabulary min word count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/test split")
    args = parser.parse_args()

    print(f"Loading SMS Spam Collection...")
    df = load_dataset()
    train_df, test_df = split_train_test(df, random_state=args.seed)
    print(f"  total: {len(df):>5}  train: {len(train_df):>5}  test: {len(test_df):>5}")

    print(f"\nBuilding vocabulary (min_count={args.min_count})...")
    vocab = Vocabulary(min_count=args.min_count).fit(train_df["text"])
    print(f"  |V| = {len(vocab):,}")

    print(f"\nFitting Multinomial Naive Bayes (alpha={args.alpha})...")
    clf = MultinomialNB(vocab=vocab, alpha=args.alpha)
    clf.fit(train_df["text"], train_df["is_spam"])

    print(f"\nEvaluating on test fold...")
    y_test = test_df["is_spam"].to_numpy()
    y_pred = clf.predict(test_df["text"])

    print()
    print(classification_report(y_test, y_pred, class_names=["ham", "spam"]))
    print()
    print(f"Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"           pred ham   pred spam")
    print(f"true ham   {cm[0,0]:>9}   {cm[0,1]:>9}")
    print(f"true spam  {cm[1,0]:>9}   {cm[1,1]:>9}")

    # Persist predictions for later analysis (gitignored).
    RUNS_DIR.mkdir(exist_ok=True)
    np.savetxt(RUNS_DIR / "predictions.csv", y_pred, fmt="%d", header="is_spam_pred", comments="")
    print(f"\nPredictions saved to {RUNS_DIR / 'predictions.csv'}")


if __name__ == "__main__":
    main()
