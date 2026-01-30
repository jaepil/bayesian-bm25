"""Standard BM25 scoring with both original and rewritten formulations."""

import math

from bayesian_bm25.math_utils import EPSILON


class BM25Scorer:
    """BM25 scorer supporting both standard and rewritten formulations.

    Parameters k1 and b control term frequency saturation and
    document length normalization respectively.
    """

    def __init__(self, corpus, k1=1.2, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b

    def idf(self, term):
        """Robertson-Sparck Jones IDF (Definition 3.1.1).

        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))
        """
        n = self.corpus.n
        df_t = self.corpus.df.get(term, 0)
        return math.log((n - df_t + 0.5) / (df_t + 0.5))

    def _length_norm(self, doc):
        """Compute the length normalization factor: 1 - b + b * dl/avgdl."""
        return 1.0 - self.b + self.b * doc["length"] / self.corpus.avgdl

    def score_term_standard(self, term, doc):
        """Standard BM25 term score (Definition 1.1.1).

        score(t, d) = IDF(t) * (k1 + 1) * tf / (k1 * norm + tf)
        """
        tf = doc["term_freq"].get(term, 0)
        if tf == 0:
            return 0.0
        norm = self._length_norm(doc)
        idf_val = self.idf(term)
        return idf_val * (self.k1 + 1.0) * tf / (self.k1 * norm + tf)

    def score_term_rewritten(self, term, doc):
        """Rewritten BM25 term score (Definition 3.2.1).

        Algebraically equivalent to standard form:
        score(t, d) = IDF(t) * boost(t, d)
        where boost(t, d) = (k1 + 1) * tf / (k1 * norm + tf)
        """
        tf = doc["term_freq"].get(term, 0)
        if tf == 0:
            return 0.0
        norm = self._length_norm(doc)
        boost = (self.k1 + 1.0) * tf / (self.k1 * norm + tf)
        idf_val = self.idf(term)
        return idf_val * boost

    def score(self, query_terms, doc):
        """Sum BM25 score over all query terms."""
        return sum(self.score_term_standard(t, doc) for t in query_terms)

    def upper_bound(self, term):
        """Upper bound on BM25 term score (Theorem 3.2.3).

        The maximum boost is (k1 + 1), so upper_bound = (k1 + 1) * IDF(t).
        Only meaningful when IDF > 0.
        """
        idf_val = self.idf(term)
        if idf_val <= 0:
            return 0.0
        return (self.k1 + 1.0) * idf_val
