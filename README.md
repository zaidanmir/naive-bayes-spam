# Naive Bayes Spam Classifier

A Multinomial Naive Bayes classifier for SMS spam detection, implemented from
first principles in NumPy. The Bayes' theorem decomposition, log-space
likelihood, and Laplace smoothing are all derived by hand in
[`notes/derivation.md`](notes/derivation.md), and the implementation matches
the derivation step by step.

A scikit-learn baseline is included for sanity checking — both implementations
agree to four decimal places on every metric.

## Results

Stratified 80/20 split (`random_state=42`), `min_count=2`, `alpha=1.0`:

| Implementation | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | \|V\| |
|---|---:|---:|---:|---:|---:|
| **From scratch (this repo)** | 0.9821 | 0.9574 | 0.9060 | **0.9310** | 3,452 |
| scikit-learn `MultinomialNB`  | 0.9821 | 0.9574 | 0.9060 | 0.9310     | 3,372 |

Confusion matrix on the 1,115-sample test fold:

```
              pred ham    pred spam
true ham      960          6
true spam     14           135
```

The two vocabulary sizes differ slightly because our `min_count` is on total
corpus frequency while scikit-learn's `min_df` is on document frequency. The
gap consists of low-frequency words that smoothing makes near-uninformative,
which is why every classification metric still matches exactly.

Most spam-indicative words learned (largest `log P(w|spam) − log P(w|ham)`):

```
claim       +5.73    pobox       +4.70
prize       +5.48    awarded     +4.67
won         +5.23    landline    +4.57
tone        +4.99    uk          +4.50
ppm         +4.92    www         +4.50
guaranteed  +4.89    ringtone    +4.42
cs          +4.87    collection  +4.38
```

These are the canonical UK SMS-spam tokens — premium-rate competitions,
ringtone scams, prize-claim phishing.

## How to run

```bash
git clone https://github.com/zaidanmir/naive-bayes-spam.git
cd naive-bayes-spam
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fit + evaluate
python train.py

# Compare against scikit-learn baseline
python -m bench.sklearn_baseline

# Run unit tests
python -m unittest discover tests
```

The dataset (UCI SMS Spam Collection, ~470 KB) is downloaded automatically on
first run by `src/data.py`.

A `Makefile` wraps the same commands: `make install`, `make train`,
`make bench`, `make test`, `make all`.

## Project structure

```
naive-bayes-spam/
├── README.md
├── LICENSE
├── requirements.txt
├── train.py                 # End-to-end CLI entrypoint
├── data/
│   └── raw/                 # Downloaded dataset (gitignored)
├── src/
│   ├── data.py              # SMS Spam loader + stratified train/test split
│   ├── vocab.py             # Tokenizer + Vocabulary class
│   ├── classifier.py        # MultinomialNB.fit / predict / predict_log_proba
│   └── evaluate.py          # accuracy, precision/recall/F1, confusion matrix
├── bench/
│   ├── sklearn_baseline.py  # Sanity-check vs sklearn.MultinomialNB
│   └── results.md           # Side-by-side metric table
├── notes/
│   └── derivation.md        # Bayes' theorem, log-likelihood, Laplace smoothing
├── tests/
│   └── test_pipeline.py     # 19 unit tests across all modules
└── runs/                    # Saved predictions (gitignored)
```

## Implementation notes

- **No autograd, no `sklearn.naive_bayes` import in the model code.** Only NumPy
  + pandas (for the data loader). scikit-learn appears in
  [`bench/sklearn_baseline.py`](bench/sklearn_baseline.py) for the baseline
  comparison only.
- **Log-space throughout.** The fit step stores `log_prior_` and
  `log_likelihood_` directly; predict computes
  `log_prior_ + count_vector @ log_likelihood_.T` in a single matmul instead
  of multiplying many small probabilities.
- **Laplace smoothing** with `alpha=1.0` by default. Smaller `alpha` underfits
  the long tail; larger `alpha` collapses predictions toward the prior. See
  [§7 of the derivation notes](notes/derivation.md) for the bias/variance tradeoff.
- **Stratified train/test split** because the prior is ~13.4% spam. Random
  splits drift the class balance and inflate variance in the test metrics.

## References

- Almeida, T.A. and Hidalgo, J.M.G. (2011). *Contributions to the Study of SMS
  Spam Filtering: New Collection and Results.* Proceedings of the 2011 ACM
  Symposium on Document Engineering — the original UCI SMS Spam Collection
  paper. [Dataset link](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval* (2008),
  Chapter 13: Text classification and Naive Bayes — the standard textbook
  treatment of the algorithm.
