# Bayesian BM25 Experimental Validation

> **Note**: This repository contains experimental code for validating the paper only. For the reference implementation, see [cognica-io/bayesian-bm25](https://github.com/cognica-io/bayesian-bm25).

Hybrid search -- combining keyword matching (BM25) with vector similarity -- is now standard practice, but the combination method is usually ad-hoc: a hand-tuned weighted sum, or Reciprocal Rank Fusion that discards score magnitudes entirely. The [Bayesian BM25 paper](https://doi.org/10.5281/zenodo.18414941) solves this by converting raw BM25 scores into calibrated probabilities via Bayes' theorem. Once both lexical and vector signals are valid probabilities, they combine through standard probability theory -- $P(A \cap B)$ for AND, $P(A \cup B)$ for OR -- with no tuning parameters and formal guarantees on the output bounds.

**The theory was originally formalized and implemented in C++ for [Cognica Database](https://cognica.io) in February 2025, and has been battle-tested against large-scale production data for nearly a year.**

This repository is a pure Python experimental validation of that paper. Ten experiments verify the core claims numerically: formula equivalence, score calibration, monotonicity, prior bounds, IDF properties, hybrid search bounds, method comparison, numerical stability, parameter learning convergence, and conjunction/disjunction bounds. All 10 pass. No external dependencies -- uses dict-based test documents with hand-crafted vector embeddings, simple string matching for lexical search, and brute-force cosine similarity for vector search.

## Quick Start

```
python -m bayesian_bm25.run_experiments
```

Expected output: 10 experiment results, each with PASS/FAIL and supporting detail. All experiments should pass.

## What This Validates

The [Bayesian BM25 paper](https://doi.org/10.5281/zenodo.18414941) reinterprets classical BM25 scoring through a Bayesian lens. Instead of producing unbounded relevance scores, it maps them to calibrated probabilities using Bayes' theorem with a sigmoid likelihood model and informative priors. This enables principled hybrid search: once both lexical (BM25) and vector (cosine similarity) scores are valid probabilities, they can be combined using probability theory (AND = joint, OR = union) rather than ad-hoc methods like RRF or linear combination.

### The 10 Experiments

| # | Experiment | Paper Reference | What It Checks |
|---|-----------|-----------------|----------------|
| 1 | BM25 Formula Equivalence | Def 1.1.1, 3.2.1 | Standard and rewritten BM25 formulas produce identical scores |
| 2 | Score Calibration | Sec 4 | All Bayesian outputs in [0,1], ranking order preserved from raw BM25 |
| 3 | Monotonicity Preservation | Thm 4.3.1 | Higher term frequency yields higher Bayesian score (same doc length) |
| 4 | Prior Bounds | Thm 4.2.4 | All composite priors stay within [0.1, 0.9] |
| 5 | IDF Properties | Thm 3.1.2, 3.1.3, 3.2.3 | IDF non-negativity, monotonic decrease with df, score upper bound |
| 6 | Hybrid Search Quality | Thm 5.1.2, 5.2.2 | AND <= min(inputs), OR >= max(inputs) on real query-document pairs |
| 7 | Naive vs RRF vs Bayesian | Thm 1.2.2, Def 1.3.1 | Side-by-side ranking comparison of five fusion methods |
| 8 | Log-space Stability | Thm 5.3.1, Def 5.3.2 | No NaN/Inf for extreme probabilities (1e-15 to 1-1e-10) and sigmoid inputs (-700 to 700) |
| 9 | Parameter Learning | Alg 8.3.1, Def 8.1.1 | Gradient descent converges: loss decreases, learned alpha > 0 |
| 10 | Conjunction/Disjunction Bounds | Thm 5.1.2, 5.2.2 | AND <= min, OR >= max, AND <= OR across 61 probability combinations |

## Package Structure

```
bayesian_bm25/
    __init__.py              Package exports
    math_utils.py            sigmoid, cosine_similarity, safe_log, clamp
    tokenizer.py             Tokenizer -- lowercase + split on non-alphanumeric
    corpus.py                Corpus -- documents + inverted index statistics
    bm25_scorer.py           BM25Scorer -- standard and rewritten formulations
    bayesian_scorer.py       BayesianBM25Scorer -- likelihood, prior, posterior
    vector_scorer.py         VectorScorer -- cosine similarity to probability
    hybrid_scorer.py         HybridScorer -- probabilistic AND/OR, naive sum, RRF
    parameter_learner.py     ParameterLearner -- gradient descent for alpha, beta
    experiments.py           ExperimentRunner -- 10 validation experiments
    run_experiments.py       Entry point with test data and main()
```

One main class per file. Dependency graph:

```
math_utils        (no deps)
tokenizer         (no deps)
corpus            -> tokenizer
bm25_scorer       -> corpus, math_utils
bayesian_scorer   -> bm25_scorer, math_utils
vector_scorer     -> math_utils
hybrid_scorer     -> bayesian_scorer, vector_scorer, math_utils
parameter_learner -> math_utils
experiments       -> all above
run_experiments   -> experiments, corpus, tokenizer
```

## Test Corpus

20 documents in 4 thematic clusters, each with an 8-dimensional hand-crafted embedding vector:

| Cluster | Documents | Topics |
|---------|-----------|--------|
| ML | d01--d05 | Machine learning, deep learning, reinforcement, transfer learning |
| IR | d06--d10 | Information retrieval, BM25, TF-IDF, query expansion, relevance feedback |
| DB | d11--d15 | SQL, NoSQL, indexing, transactions, distributed databases |
| Cross-cutting | d16--d20 | Vector search, hybrid search, Bayesian probability, cosine similarity |

Embedding dimensions: `[ML, DL/neural, IR/search, ranking, DB, distributed, probability, vectors]`

7 queries covering common multi-term, rare terms, single term, hybrid, and multi-term specific match scenarios. Each query includes pre-tokenized terms, an embedding vector, and a list of relevant document IDs.

## Experiment Results

All 10 experiments pass. Corpus: 20 documents, avgdl=10.8, vocabulary=154 terms, 7 queries.

| # | Experiment | Result | Key Observation |
|---|-----------|--------|-----------------|
| 1 | BM25 Formula Equivalence | PASS | max_diff=$4.44 \times 10^{-16}$ across 280 comparisons. The standard form and rewritten form are identical to floating-point epsilon, confirming the algebraic equivalence proof. |
| 2 | Score Calibration | PASS | Every Bayesian output falls in $[0, 1]$ and the ranking order from raw BM25 is fully preserved. The sigmoid-based Bayesian transformation is a monotonic mapping that does not distort relative document ordering. |
| 3 | Monotonicity Preservation | PASS | Synthetic test confirms that the posterior is strictly monotonic in the raw BM25 score for any fixed prior. Higher term frequency always produces a higher relevance probability. |
| 4 | Prior Bounds | PASS | Observed prior range $[0.3146, 0.4126]$, well within the theoretical $[0.1, 0.9]$. The composite prior is conservative: moderate TF values and near-average document lengths keep priors near 0.35, preventing the prior from dominating the posterior. |
| 5 | IDF Properties | PASS | IDF is non-negative for all terms with $df \le N/2$, monotonically decreases as document frequency rises, and every actual term score stays below the $(k_1 + 1) \cdot \text{IDF}$ upper bound. |
| 6 | Hybrid Search Quality | PASS | 140 query-document pairs tested. Probabilistic AND never exceeds $\min(P_{\text{lex}}, P_{\text{vec}})$ and probabilistic OR never falls below $\max(P_{\text{lex}}, P_{\text{vec}})$. The independence-assumption bounds hold without exception. |
| 7 | Naive vs RRF vs Bayesian | PASS | The five methods (BM25, Bayesian, hybrid OR, naive sum, RRF) produce different top-5 rankings for 7 queries. Hybrid OR surfaces semantically relevant documents (e.g., d20 for "vector search embeddings") that pure BM25 misses, while naive sum and RRF lack this capability due to scale mismatch or rank-only information. |
| 8 | Log-space Stability | PASS | 26 tests with probabilities from $10^{-15}$ to $1 - 10^{-10}$ and sigmoid inputs from $-700$ to $700$. No NaN, no Inf, all results in $[0, 1]$. Log-space computation and the numerically stable sigmoid eliminate catastrophic cancellation and overflow. |
| 9 | Parameter Learning | PASS | Gradient descent learned $\alpha = 4.3138$, $\beta = 0.7431$. Cross-entropy loss dropped from $0.5750$ to $0.0626$ (89% reduction) with all 500 steps monotonically decreasing. Positive $\alpha$ confirms that higher BM25 scores correlate with relevance. The steep $\alpha$ indicates a sharp decision boundary: the sigmoid transitions rapidly around the learned threshold $\beta$. |
| 10 | Conjunction/Disjunction Bounds | PASS | 61 probability combinations (49 pairs + 12 triples) all satisfy $P(A \cap B) \le \min(P(A), P(B))$, $P(A \cup B) \ge \max(P(A), P(B))$, and $P(A \cap B) \le P(A \cup B)$. These are necessary properties for any valid probabilistic fusion framework. |

## Key Formulas Implemented

**BM25 (Definition 1.1.1)**

$$
\text{score}(t, d) = \text{IDF}(t) \cdot \frac{(k_1 + 1) \cdot tf}{k_1 \cdot (1 - b + b \cdot dl / avgdl) + tf}
$$

**Robertson-Sparck Jones IDF (Definition 3.1.1)**

$$
\text{IDF}(t) = \log \frac{N - df(t) + 0.5}{df(t) + 0.5}
$$

**Bayesian Posterior (Theorem 4.1.3)**

$$
P(\text{rel} \mid s) = \frac{L \cdot p}{L \cdot p + (1 - L)(1 - p)}
$$

where $L = \sigma(\alpha \cdot (s - \beta))$ is the likelihood and $p$ is the composite prior combining term frequency and document length signals.

**Probabilistic AND (Theorem 5.1.1)**

$$
P(A \cap B) = P(A) \cdot P(B)
$$

Assumes independence. Computed in log-space for numerical stability.

**Probabilistic OR (Theorem 5.2.1)**

$$
P(A \cup B) = 1 - (1 - P(A))(1 - P(B))
$$

Computed in log-space for numerical stability.

**Cross-Entropy Loss (Definition 8.1.1)**

$$
\mathcal{L} = -\frac{1}{N} \sum_{i} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]
$$

where $p = \sigma(\alpha \cdot (s - \beta))$, optimized via gradient descent (Algorithm 8.3.1) to learn calibration parameters $\alpha$ and $\beta$.

## Requirements

Python 3.6+ with only the standard library (`math`, `re`).
