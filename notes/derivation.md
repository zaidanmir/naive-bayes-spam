# Naive Bayes — derivation

These notes derive the model implemented in `src/classifier.py` from first
principles. Every formula here corresponds to a line in the code.

## 1. Goal

Given a text $x$ (a sequence of words), predict its class $c \in \mathcal{C}$.
We want the **maximum-a-posteriori** prediction:

$$
\hat c = \arg\max_{c \in \mathcal{C}} P(c \mid x)
$$

For SMS spam classification $\mathcal{C} = \{\text{ham}, \text{spam}\}$.

## 2. Bayes' theorem

By Bayes' rule,

$$
P(c \mid x) = \frac{P(x \mid c) \, P(c)}{P(x)}
$$

The denominator $P(x)$ is constant in $c$ (it doesn't depend on which class
we're considering — same evidence either way), so it can be dropped from the
arg-max:

$$
\hat c = \arg\max_c \, P(x \mid c) \, P(c)
$$

Two terms remain:

- **Class prior** $P(c)$ — how common is class $c$ in general?
- **Class-conditional likelihood** $P(x \mid c)$ — how likely is text $x$
  given that the class is $c$?

## 3. The "Naive" conditional independence assumption

For text classification, $x$ is a bag of words. Modelling the joint
distribution of an entire sentence is intractable — there are exponentially
many possible word sequences. Naive Bayes makes the **conditional
independence assumption**: given the class, words are independent of each
other. So:

$$
P(x \mid c) = \prod_w P(w \mid c)^{x_w}
$$

where $x_w$ is the count of word $w$ in document $x$.

This assumption is obviously false — the word *win* is more likely to follow
*you* than *the*. But the approximation works remarkably well for spam
classification because the model only needs the right *rankings* of
$P(c \mid x)$ across classes, not the right *values*. Badly calibrated but
correctly ordered probabilities still give the correct arg-max.

The "Multinomial" qualifier refers specifically to this counts-based model
(as opposed to, say, Bernoulli Naive Bayes which uses presence/absence).

## 4. Log-space transformation

Multiplying many small probabilities causes numerical underflow in
`float64` after a few hundred terms. The fix is to take logs:

$$
\log P(c \mid x) \, \propto \, \log P(c) + \sum_w x_w \, \log P(w \mid c)
$$

Two reasons this is the right move:

1. **Argmax is invariant under monotonic transforms.** $\arg\max f(c) = \arg\max \log f(c)$.
2. **Sums replace products.** Numerically stable; small probabilities become
   manageable negative numbers instead of vanishing toward zero.

The $\propto$ symbol just acknowledges we've dropped the constant $\log P(x)$
term — it doesn't affect the arg-max.

In code (`predict_log_proba`):

```python
return self.log_prior_ + encoded @ self.log_likelihood_.T
```

The matmul is the vectorised version of $\sum_w x_w \log P(w \mid c)$ across
all texts and all classes simultaneously.

## 5. Estimating the parameters

Given a labelled training corpus, the maximum-likelihood estimates are:

**Class prior:**

$$
\hat P(c) = \frac{\#\{i : y_i = c\}}{N}
$$

— class $c$'s share of the training corpus. In log space:
`log_prior_ = log(class_counts / N)`.

**Class-conditional word probabilities:**

$$
\hat P(w \mid c) = \frac{n_{c,w}}{\sum_{w'} n_{c,w'}}
$$

where $n_{c,w}$ is the total number of times word $w$ appears across
documents of class $c$.

## 6. The zero-frequency problem

The MLE estimate above assigns probability **zero** to any word that didn't
appear in class $c$'s training documents. After taking the log,
$\log 0 = -\infty$, and a single $-\infty$ term destroys the entire score:

$$
\log P(c \mid x) = \log P(c) + \cdots + (-\infty) + \cdots = -\infty
$$

So *one* unseen word makes class $c$ infinitely improbable, regardless of
how strongly all the other words point to it. That's clearly wrong.

## 7. Laplace (additive) smoothing

The standard fix is to add a positive constant $\alpha$ to every count:

$$
\hat P(w \mid c) = \frac{n_{c,w} + \alpha}{\sum_{w'} n_{c,w'} + \alpha |V|}
$$

where $|V|$ is the vocabulary size.

The numerator is straightforward — every word now has at least $\alpha$
"pseudo-counts." The $\alpha |V|$ in the denominator falls out of the
constraint that probabilities must sum to 1:

$$
\sum_w \hat P(w \mid c) = \frac{\sum_w (n_{c,w} + \alpha)}{\sum_{w'} n_{c,w'} + \alpha |V|}
= \frac{(\sum_w n_{c,w}) + \alpha |V|}{\sum_{w'} n_{c,w'} + \alpha |V|} = 1
$$

In code (`fit`):

```python
smoothed = word_counts + self.alpha
self.log_likelihood_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
```

`smoothed.sum(axis=1)` gives $\sum_{w'} (n_{c,w'} + \alpha) = \sum n_{c,w'} + \alpha|V|$
in one numpy call — same value, never explicitly forming the $\alpha|V|$ term.

### Choosing $\alpha$

- $\alpha = 1$ — **Laplace smoothing**. The most common default. Equivalent
  to a uniform prior over word distributions.
- $\alpha = 0.5$ — Jeffreys prior, occasionally used.
- $\alpha \to 0^+$ — approaches the unsmoothed MLE. Bad if there are
  any unseen words at test time.
- $\alpha \to \infty$ — predictions collapse toward the class prior; the
  data is ignored.

In practice $\alpha = 1$ works well; it can be tuned on a validation set
if the dataset is large enough to make the difference detectable.

## 8. Summary — the prediction rule

Putting it all together:

$$
\hat c = \arg\max_c \left[ \log \hat P(c) + \sum_w x_w \log \hat P(w \mid c) \right]
$$

with $\hat P(c)$ and $\hat P(w \mid c)$ the smoothed estimates from the
training corpus.

That's the entire model. Three fitted-parameter arrays —
`classes_`, `log_prior_`, `log_likelihood_` — and one matmul at predict
time. No iterative optimisation, no learning rate, no convergence questions.

## 9. Why it works on SMS spam

Spam messages are lexically distinctive: words like *claim*, *prize*, *won*,
*ringtone*, *guaranteed* almost never appear in legitimate ham messages, and
ham contains a long tail of conversational vocabulary that almost never
appears in spam. The conditional independence assumption is a good
approximation in this regime because *individual word presence* — not word
order or syntactic structure — carries most of the signal. On the UCI SMS
Spam Collection, this implementation reaches **0.9821 accuracy / 0.9310 F1
(spam class)** on a stratified 80/20 split with `min_count=2, alpha=1.0`.
