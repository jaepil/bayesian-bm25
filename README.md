# Bayesian BM25 Experimental Validation

Pure Python experimental code that validates the mathematical claims in the [Bayesian BM25 paper](https://doi.org/10.5281/zenodo.18414941). No external dependencies -- uses dict-based test documents with hand-crafted vector embeddings, simple string matching for lexical search, and brute-force cosine similarity for vector search.

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
