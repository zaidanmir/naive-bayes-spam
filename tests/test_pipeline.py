"""Unit tests for tokenizer, vocabulary, classifier, and metrics.

Run with: python -m unittest discover tests
"""
from __future__ import annotations

import unittest

import numpy as np

from src.classifier import MultinomialNB
from src.evaluate import accuracy, confusion_matrix, precision_recall_f1
from src.vocab import Vocabulary, tokenize


class TestTokenize(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(tokenize("Hello WORLD"), ["hello", "world"])

    def test_strips_punctuation(self):
        self.assertEqual(tokenize("hi, are you free? yes!"), ["hi", "are", "you", "free", "yes"])

    def test_strips_digits_and_currency(self):
        self.assertEqual(tokenize("call 0800-555 for £10 off"), ["call", "for", "off"])

    def test_keeps_apostrophes(self):
        self.assertEqual(tokenize("don't won't can't"), ["don't", "won't", "can't"])

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])


class TestVocabulary(unittest.TestCase):
    def test_min_count_1_keeps_everything(self):
        vocab = Vocabulary(min_count=1).fit(["the cat", "the dog"])
        self.assertEqual(set(vocab.idx_to_word), {"the", "cat", "dog"})

    def test_min_count_drops_singletons(self):
        vocab = Vocabulary(min_count=2).fit(["the cat sat", "the dog ran", "the cat ran"])
        # 'the' appears 3 times, 'cat' twice, 'ran' twice -> kept; 'sat'/'dog' once -> dropped
        self.assertEqual(set(vocab.idx_to_word), {"the", "cat", "ran"})

    def test_encode_returns_correct_counts(self):
        vocab = Vocabulary(min_count=1).fit(["hi hi hi bye"])
        counts = vocab.encode("hi hi bye unknown")
        self.assertEqual(counts.sum(), 3)  # "unknown" not in vocab
        self.assertEqual(counts[vocab.word_to_idx["hi"]], 2)
        self.assertEqual(counts[vocab.word_to_idx["bye"]], 1)

    def test_encode_unknown_words_dropped(self):
        vocab = Vocabulary(min_count=1).fit(["hello world"])
        counts = vocab.encode("foo bar baz")
        self.assertEqual(counts.sum(), 0)


class TestClassifier(unittest.TestCase):
    def setUp(self):
        # Trivially separable corpus: spam contains 'win prize' tokens, ham contains 'meet lunch'.
        self.train_texts = [
            "win prize click now",
            "win cash free prize",
            "free win now claim",
            "lunch meeting tomorrow",
            "meet at the cafe",
            "let us have lunch",
        ]
        self.train_labels = np.array([1, 1, 1, 0, 0, 0])
        self.vocab = Vocabulary(min_count=1).fit(self.train_texts)
        self.clf = MultinomialNB(vocab=self.vocab, alpha=1.0).fit(
            self.train_texts, self.train_labels
        )

    def test_classifier_memorises_training_data(self):
        # On separable data with no smoothing pressure, the classifier should
        # reproduce its training labels exactly.
        preds = self.clf.predict(self.train_texts)
        np.testing.assert_array_equal(preds, self.train_labels)

    def test_classifier_handles_unseen_words(self):
        # No -inf in predicted log-probabilities even when text contains
        # words not in the vocabulary.
        log_proba = self.clf.predict_log_proba(["completely unknown vocabulary asdfghjkl"])
        self.assertTrue(np.all(np.isfinite(log_proba)))

    def test_classifier_log_prior_sums_to_one(self):
        # exp of log priors should sum to 1 (probability distribution).
        self.assertAlmostEqual(float(np.exp(self.clf.log_prior_).sum()), 1.0, places=6)

    def test_classifier_log_likelihood_rows_sum_to_one(self):
        # Each row of log_likelihood is a probability distribution over the vocabulary.
        row_sums = np.exp(self.clf.log_likelihood_).sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-6)

    def test_classifier_alpha_zero_rejected(self):
        with self.assertRaises(ValueError):
            MultinomialNB(vocab=self.vocab, alpha=0)


class TestEvaluate(unittest.TestCase):
    def test_accuracy_perfect(self):
        self.assertEqual(accuracy([0, 1, 0], [0, 1, 0]), 1.0)

    def test_accuracy_zero(self):
        self.assertEqual(accuracy([0, 1, 0], [1, 0, 1]), 0.0)

    def test_precision_recall_f1_hand_checked(self):
        # 5 ham + 3 spam, predictions: 1 spam misclassified, 1 ham misclassified.
        # spam: TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3.
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 1])
        p, r, f1 = precision_recall_f1(y_true, y_pred, positive_class=1)
        self.assertAlmostEqual(p, 2 / 3, places=6)
        self.assertAlmostEqual(r, 2 / 3, places=6)
        self.assertAlmostEqual(f1, 2 / 3, places=6)

    def test_precision_zero_division(self):
        # No predicted positives -> precision should be 0, not raise.
        p, r, f1 = precision_recall_f1([0, 0, 1], [0, 0, 0], positive_class=1)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f1, 0.0)

    def test_confusion_matrix_shape_and_counts(self):
        cm = confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1])
        np.testing.assert_array_equal(cm, [[1, 1], [1, 1]])


if __name__ == "__main__":
    unittest.main()
