"""Entry point: test corpus, queries, and experiment execution."""

import math

from bayesian_bm25.tokenizer import Tokenizer
from bayesian_bm25.corpus import Corpus
from bayesian_bm25.experiments import ExperimentRunner


# --------------------------------------------------------------------------
# Test corpus: 20 documents in 4 clusters with 8-dimensional embeddings
# --------------------------------------------------------------------------

# Embedding conventions (8 dimensions):
#   [ML, DL/neural, IR/search, ranking, DB, distributed, probability, vectors]

DOCUMENTS = [
    # ML cluster (d01-d05): strong ML/DL signals
    {
        "id": "d01",
        "text": "Machine learning algorithms learn patterns from data using statistical methods",
        "embedding": [0.9, 0.3, 0.1, 0.0, 0.1, 0.0, 0.4, 0.1],
    },
    {
        "id": "d02",
        "text": "Deep learning neural networks require large training datasets for supervised learning",
        "embedding": [0.8, 0.9, 0.1, 0.0, 0.0, 0.1, 0.2, 0.3],
    },
    {
        "id": "d03",
        "text": "Unsupervised learning discovers hidden structure in unlabeled data",
        "embedding": [0.9, 0.4, 0.0, 0.0, 0.1, 0.0, 0.3, 0.2],
    },
    {
        "id": "d04",
        "text": "Reinforcement learning agents maximize cumulative reward through exploration",
        "embedding": [0.8, 0.5, 0.0, 0.0, 0.0, 0.1, 0.3, 0.0],
    },
    {
        "id": "d05",
        "text": "Transfer learning adapts pre-trained models to new domains with limited data",
        "embedding": [0.9, 0.7, 0.1, 0.0, 0.0, 0.0, 0.2, 0.3],
    },
    # IR cluster (d06-d10): strong IR/search/ranking signals
    {
        "id": "d06",
        "text": "Information retrieval systems search and rank documents by relevance to queries",
        "embedding": [0.1, 0.0, 0.9, 0.8, 0.0, 0.0, 0.2, 0.1],
    },
    {
        "id": "d07",
        "text": "BM25 is a bag of words retrieval function that ranks documents based on term frequency",
        "embedding": [0.1, 0.0, 0.8, 0.9, 0.0, 0.0, 0.3, 0.0],
    },
    {
        "id": "d08",
        "text": "TF-IDF weighting reflects how important a word is to a document in a collection",
        "embedding": [0.1, 0.0, 0.8, 0.7, 0.0, 0.0, 0.2, 0.0],
    },
    {
        "id": "d09",
        "text": "Query expansion improves search recall by adding related terms to the original query",
        "embedding": [0.2, 0.0, 0.9, 0.6, 0.0, 0.0, 0.1, 0.1],
    },
    {
        "id": "d10",
        "text": "Relevance feedback uses explicit user judgments to improve retrieval performance",
        "embedding": [0.2, 0.0, 0.8, 0.7, 0.0, 0.0, 0.2, 0.0],
    },
    # DB cluster (d11-d15): strong DB signals
    {
        "id": "d11",
        "text": "Relational databases store data in tables with SQL as the query language",
        "embedding": [0.0, 0.0, 0.1, 0.0, 0.9, 0.2, 0.0, 0.0],
    },
    {
        "id": "d12",
        "text": "NoSQL databases provide flexible schema design for unstructured data storage",
        "embedding": [0.0, 0.0, 0.1, 0.0, 0.9, 0.3, 0.0, 0.0],
    },
    {
        "id": "d13",
        "text": "Database indexing structures like B-trees accelerate data retrieval operations",
        "embedding": [0.0, 0.0, 0.3, 0.1, 0.9, 0.1, 0.0, 0.0],
    },
    {
        "id": "d14",
        "text": "Transaction processing ensures ACID properties for reliable data operations",
        "embedding": [0.0, 0.0, 0.0, 0.0, 0.9, 0.3, 0.0, 0.0],
    },
    {
        "id": "d15",
        "text": "Distributed databases partition data across multiple nodes for scalability",
        "embedding": [0.0, 0.0, 0.1, 0.0, 0.8, 0.9, 0.0, 0.0],
    },
    # Cross-cutting cluster (d16-d20): mixed signals
    {
        "id": "d16",
        "text": "Vector search uses embedding similarity to find semantically related documents",
        "embedding": [0.3, 0.3, 0.7, 0.5, 0.1, 0.0, 0.2, 0.9],
    },
    {
        "id": "d17",
        "text": "Hybrid search combines lexical matching with vector similarity for better retrieval",
        "embedding": [0.2, 0.2, 0.8, 0.6, 0.0, 0.0, 0.3, 0.8],
    },
    {
        "id": "d18",
        "text": "Bayesian probability provides a framework for updating beliefs with new evidence",
        "embedding": [0.3, 0.1, 0.2, 0.2, 0.0, 0.0, 0.9, 0.1],
    },
    {
        "id": "d19",
        "text": "Probabilistic models estimate relevance scores using statistical inference methods",
        "embedding": [0.4, 0.1, 0.5, 0.4, 0.0, 0.0, 0.8, 0.2],
    },
    {
        "id": "d20",
        "text": "Cosine similarity measures the angle between two vectors in high-dimensional space",
        "embedding": [0.2, 0.1, 0.3, 0.2, 0.0, 0.0, 0.3, 0.9],
    },
]


# --------------------------------------------------------------------------
# Queries with terms, embeddings, and relevance judgments
# --------------------------------------------------------------------------

QUERIES = [
    {
        "text": "machine learning",
        "terms": ["machine", "learning"],
        "embedding": [0.9, 0.5, 0.1, 0.0, 0.0, 0.0, 0.3, 0.2],
        "relevant": ["d01", "d02", "d03", "d04", "d05"],
    },
    {
        "text": "Bayesian probability",
        "terms": ["bayesian", "probability"],
        "embedding": [0.3, 0.1, 0.2, 0.2, 0.0, 0.0, 0.9, 0.1],
        "relevant": ["d18", "d19"],
    },
    {
        "text": "search",
        "terms": ["search"],
        "embedding": [0.1, 0.0, 0.9, 0.6, 0.0, 0.0, 0.1, 0.3],
        "relevant": ["d06", "d09", "d16", "d17"],
    },
    {
        "text": "transaction processing",
        "terms": ["transaction", "processing"],
        "embedding": [0.0, 0.0, 0.0, 0.0, 0.9, 0.3, 0.0, 0.0],
        "relevant": ["d14"],
    },
    {
        "text": "data",
        "terms": ["data"],
        "embedding": [0.4, 0.2, 0.3, 0.1, 0.4, 0.2, 0.2, 0.2],
        "relevant": ["d01", "d03", "d05", "d11", "d12", "d13", "d14", "d15"],
    },
    {
        "text": "vector search embeddings",
        "terms": ["vector", "search", "embeddings"],
        "embedding": [0.2, 0.2, 0.7, 0.4, 0.0, 0.0, 0.2, 0.9],
        "relevant": ["d16", "d17", "d20"],
    },
    {
        "text": "retrieval augmented generation",
        "terms": ["retrieval", "augmented", "generation"],
        "embedding": [0.4, 0.4, 0.7, 0.5, 0.0, 0.0, 0.2, 0.4],
        "relevant": ["d06", "d07", "d10", "d17"],
    },
]


def build_corpus():
    """Build the test corpus from the document definitions."""
    tokenizer = Tokenizer()
    corpus = Corpus(tokenizer)
    for doc_def in DOCUMENTS:
        corpus.add_document(doc_def["id"], doc_def["text"], doc_def["embedding"])
    corpus.build_index()
    return corpus


def main():
    """Run all experiments and print results."""
    corpus = build_corpus()
    runner = ExperimentRunner(corpus, QUERIES)
    results = runner.run_all()

    print("=" * 72)
    print("Bayesian BM25 Experimental Validation")
    print("=" * 72)
    print()
    print("Corpus: %d documents, avgdl=%.1f, vocabulary=%d terms" % (
        corpus.n, corpus.avgdl, len(corpus.df)
    ))
    print("Queries: %d" % len(QUERIES))
    print()

    all_passed = True
    for name, passed, details in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print("-" * 72)
        print("[%s] %s" % (status, name))
        # Print details, indenting multi-line output
        for line in details.split("\n"):
            print("       %s" % line)
        print()

    print("=" * 72)
    if all_passed:
        print("All 10 experiments PASSED.")
    else:
        failed = [name for name, passed, _ in results if not passed]
        print("FAILED experiments: %s" % ", ".join(failed))
    print("=" * 72)


if __name__ == "__main__":
    main()
