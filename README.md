# Naive Bayes Spam Classifier

A from-scratch implementation of a Multinomial Naive Bayes classifier for SMS spam
detection, accompanied by a probabilistic write-up covering Bayes' theorem,
log-likelihood computation, and Laplace smoothing.

The point of building this from first principles (rather than calling
`sklearn.naive_bayes.MultinomialNB`) is to make the maths explicit — every
probability and gradient you'd otherwise hide behind a library call is written out.

## Status

Work in progress. See commits for granular progress.

## Planned components

- `src/data.py` — SMS Spam Collection loader and train/test split
- `src/vocab.py` — vocabulary builder and word-frequency counts
- `src/classifier.py` — Multinomial Naive Bayes (fit / predict) with Laplace smoothing
- `src/evaluate.py` — precision / recall / F1 / confusion matrix
- `notebooks/derivation.ipynb` — Bayes' theorem and log-likelihood derivation
- `train.py` — end-to-end training and evaluation entrypoint
- Comparison against `sklearn` baseline

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

## Dataset

[SMS Spam Collection (UCI)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
— 5,574 labelled SMS messages.
