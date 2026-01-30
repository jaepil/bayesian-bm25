"""Bayesian BM25 scorer that converts BM25 scores to calibrated probabilities."""

from bayesian_bm25.math_utils import sigmoid, safe_log, safe_prob, clamp


class BayesianBM25Scorer:
    """Bayesian interpretation of BM25 using likelihood and prior.

    Converts raw BM25 scores into posterior probabilities of relevance
    using Bayes' theorem with a sigmoid likelihood model.
    """

    def __init__(self, bm25_scorer, alpha=1.0, beta=0.5):
        self.bm25 = bm25_scorer
        self.alpha = alpha
        self.beta = beta

    def likelihood(self, score):
        """Likelihood via sigmoid mapping (Definition 4.1.1).

        L(s) = sigmoid(alpha * (s - beta))
        """
        return sigmoid(self.alpha * (score - self.beta))

    def tf_prior(self, tf):
        """Term-frequency based prior (Definition 4.2.1).

        P_tf = 0.2 + 0.7 * min(1, tf / 10)
        """
        return 0.2 + 0.7 * min(1.0, tf / 10.0)

    def norm_prior(self, doc_length, avg_doc_length):
        """Document length normalization prior (Definition 4.2.2).

        Shorter documents that match get higher prior. Uses a ratio-based
        formula clamped to [0.1, 0.9].
        """
        if avg_doc_length < 1.0:
            return 0.5
        ratio = doc_length / avg_doc_length
        # Documents near average length get prior ~0.5
        # Shorter documents get higher prior, longer get lower
        prior = 1.0 / (1.0 + ratio)
        return clamp(prior, 0.1, 0.9)

    def composite_prior(self, tf, doc_length, avg_doc_length):
        """Composite prior combining TF and length priors (Definition 4.2.3).

        P = clamp(0.7 * P_tf + 0.3 * P_norm, 0.1, 0.9)
        """
        p_tf = self.tf_prior(tf)
        p_norm = self.norm_prior(doc_length, avg_doc_length)
        combined = 0.7 * p_tf + 0.3 * p_norm
        return clamp(combined, 0.1, 0.9)

    def posterior(self, score, prior):
        """Bayesian posterior via Bayes' theorem (Theorem 4.1.3).

        P(rel|s) = (L * p) / (L * p + (1 - L) * (1 - p))
        """
        lik = self.likelihood(score)
        lik = safe_prob(lik)
        prior = safe_prob(prior)
        numerator = lik * prior
        denominator = numerator + (1.0 - lik) * (1.0 - prior)
        return numerator / denominator

    def score_term(self, term, doc):
        """Per-term Bayesian probability of relevance."""
        raw_score = self.bm25.score_term_standard(term, doc)
        if raw_score == 0.0:
            return 0.0
        tf = doc["term_freq"].get(term, 0)
        prior = self.composite_prior(
            tf, doc["length"], self.bm25.corpus.avgdl
        )
        return self.posterior(raw_score, prior)

    def score(self, query_terms, doc):
        """Multi-term Bayesian score using noisy-OR combination (log-space).

        P(rel|q) = 1 - prod(1 - P(rel|t_i)) for each query term t_i.
        Computed in log-space for numerical stability.
        """
        log_complement_sum = 0.0
        has_match = False
        for term in query_terms:
            p = self.score_term(term, doc)
            if p > 0.0:
                has_match = True
                p = safe_prob(p)
                log_complement_sum += safe_log(1.0 - p)
        if not has_match:
            return 0.0
        import math
        return 1.0 - math.exp(log_complement_sum)
