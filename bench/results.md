# Benchmark — from-scratch vs scikit-learn

Both runs use the same data, the same tokenizer (`src/vocab.tokenize`),
the same vocabulary cutoff (`min_count=2`), and the same Laplace
smoothing (`alpha=1.0`). Only the classifier implementation differs.

| Implementation | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | |V| |
|---|---|---|---|---|---|
| From-scratch (this repo) | 0.9821 | 0.9574 | 0.9060 | 0.9310 | 3,452 |
| scikit-learn `MultinomialNB` | 0.9821 | 0.9574 | 0.9060 | 0.9310 | 3,372 |

Accuracy delta: **0.0000**.

Reproduce: `python -m bench.sklearn_baseline`.