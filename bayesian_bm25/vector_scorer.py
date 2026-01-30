"""Vector similarity scorer using brute-force cosine similarity."""

from bayesian_bm25.math_utils import cosine_similarity, clamp


class VectorScorer:
    """Scores documents by cosine similarity to a query embedding.

    Converts raw cosine similarity [-1, 1] to a probability [0, 1]
    using Definition 7.1.2.
    """

    def score_to_probability(self, sim):
        """Convert cosine similarity to probability (Definition 7.1.2).

        P = (1 + sim) / 2, mapping [-1, 1] to [0, 1].
        """
        return clamp((1.0 + sim) / 2.0, 0.0, 1.0)

    def score(self, query_embedding, doc):
        """Compute relevance probability from query and document embeddings."""
        sim = cosine_similarity(query_embedding, doc["embedding"])
        return self.score_to_probability(sim)
