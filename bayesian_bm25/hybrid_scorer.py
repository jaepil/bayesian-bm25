"""Hybrid scorer combining Bayesian BM25 and vector search probabilistically."""

import math

from bayesian_bm25.math_utils import safe_log, safe_prob, EPSILON


class HybridScorer:
    """Combines Bayesian BM25 and vector scores using probability theory.

    Supports probabilistic AND (joint) and OR (union) combinations,
    plus baseline methods (naive sum, RRF) for comparison.
    """

    def __init__(self, bayesian_scorer, vector_scorer):
        self.bayesian = bayesian_scorer
        self.vector = vector_scorer

    def probabilistic_and(self, probs):
        """Probabilistic AND assuming independence (Theorem 5.1.1).

        P(A and B) = P(A) * P(B), computed in log-space (Definition 5.1.2).
        """
        log_sum = 0.0
        for p in probs:
            p = safe_prob(p)
            log_sum += safe_log(p)
        return math.exp(log_sum)

    def probabilistic_or(self, probs):
        """Probabilistic OR assuming independence (Theorem 5.2.1).

        P(A or B) = 1 - (1-P(A))*(1-P(B)), computed in log-space
        (Definition 5.2.2).
        """
        log_complement_sum = 0.0
        for p in probs:
            p = safe_prob(p)
            log_complement_sum += safe_log(1.0 - p)
        return 1.0 - math.exp(log_complement_sum)

    def score_and(self, query_terms, query_embedding, doc):
        """Hybrid AND: document must be relevant by both signals (Def 7.2.1)."""
        bayesian_prob = self.bayesian.score(query_terms, doc)
        vector_prob = self.vector.score(query_embedding, doc)
        if bayesian_prob < EPSILON and vector_prob < EPSILON:
            return 0.0
        return self.probabilistic_and([bayesian_prob, vector_prob])

    def score_or(self, query_terms, query_embedding, doc):
        """Hybrid OR: document relevant by either signal (Def 7.2.2)."""
        bayesian_prob = self.bayesian.score(query_terms, doc)
        vector_prob = self.vector.score(query_embedding, doc)
        return self.probabilistic_or([bayesian_prob, vector_prob])

    def naive_sum(self, scores):
        """Naive linear combination baseline (Theorem 1.2.2).

        Simple sum of scores -- not a valid probability.
        """
        return sum(scores)

    def rrf_score(self, ranks, k=60):
        """Reciprocal Rank Fusion baseline (Definition 1.3.1).

        RRF(d) = sum(1 / (k + rank_i)) for each ranking.
        """
        return sum(1.0 / (k + r) for r in ranks)
