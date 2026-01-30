"""Document corpus with inverted index for BM25 scoring."""

from bayesian_bm25.tokenizer import Tokenizer


class Corpus:
    """Stores documents and builds statistics needed for BM25.

    Each document is a dict with keys:
        id, text, embedding, tokens, length, term_freq
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or Tokenizer()
        self.documents = []
        self._doc_by_id = {}
        # Index statistics (populated by build_index)
        self.n = 0
        self.avgdl = 0.0
        self.df = {}  # term -> document frequency

    def add_document(self, doc_id, text, embedding):
        """Tokenize text and store as a document dict."""
        tokens = self.tokenizer.tokenize(text)
        term_freq = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        doc = {
            "id": doc_id,
            "text": text,
            "embedding": embedding,
            "tokens": tokens,
            "length": len(tokens),
            "term_freq": term_freq,
        }
        self.documents.append(doc)
        self._doc_by_id[doc_id] = doc

    def build_index(self):
        """Compute N, df(t) for each term, and avgdl."""
        self.n = len(self.documents)
        self.df = {}
        total_length = 0
        for doc in self.documents:
            total_length += doc["length"]
            for term in doc["term_freq"]:
                self.df[term] = self.df.get(term, 0) + 1
        self.avgdl = total_length / self.n if self.n > 0 else 0.0

    def get_document(self, doc_id):
        """Look up a document by ID."""
        return self._doc_by_id[doc_id]
