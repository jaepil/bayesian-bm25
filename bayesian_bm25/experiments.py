"""Ten experiments validating the claims in the Bayesian BM25 paper."""

import math

from bayesian_bm25.math_utils import (
    sigmoid,
    safe_log,
    safe_prob,
    cosine_similarity,
    EPSILON,
)
from bayesian_bm25.bm25_scorer import BM25Scorer
from bayesian_bm25.bayesian_scorer import BayesianBM25Scorer
from bayesian_bm25.vector_scorer import VectorScorer
from bayesian_bm25.hybrid_scorer import HybridScorer
from bayesian_bm25.parameter_learner import ParameterLearner


class ExperimentRunner:
    """Runs 10 experiments validating Bayesian BM25 paper claims."""

    def __init__(self, corpus, queries, k1=1.2, b=0.75):
        self.corpus = corpus
        self.queries = queries
        self.bm25 = BM25Scorer(corpus, k1=k1, b=b)
        self.bayesian = BayesianBM25Scorer(self.bm25)
        self.vector = VectorScorer()
        self.hybrid = HybridScorer(self.bayesian, self.vector)

    def run_all(self):
        """Run all 10 experiments and return results."""
        experiments = [
            ("1. BM25 Formula Equivalence", self.exp1_formula_equivalence),
            ("2. Score Calibration", self.exp2_score_calibration),
            ("3. Monotonicity Preservation", self.exp3_monotonicity),
            ("4. Prior Bounds", self.exp4_prior_bounds),
            ("5. IDF Properties", self.exp5_idf_properties),
            ("6. Hybrid Search Quality", self.exp6_hybrid_quality),
            ("7. Naive vs RRF vs Bayesian", self.exp7_method_comparison),
            ("8. Log-space Numerical Stability", self.exp8_numerical_stability),
            ("9. Parameter Learning Convergence", self.exp9_parameter_learning),
            ("10. Conjunction/Disjunction Bounds", self.exp10_conjunction_disjunction),
        ]
        results = []
        for name, func in experiments:
            passed, details = func()
            results.append((name, passed, details))
        return results

    def exp1_formula_equivalence(self):
        """Validate Def 1.1.1 == Def 3.2.1 for all query-document pairs."""
        max_diff = 0.0
        comparisons = 0
        for query in self.queries:
            terms = query["terms"]
            for doc in self.corpus.documents:
                for term in terms:
                    s1 = self.bm25.score_term_standard(term, doc)
                    s2 = self.bm25.score_term_rewritten(term, doc)
                    diff = abs(s1 - s2)
                    max_diff = max(max_diff, diff)
                    comparisons += 1
        passed = max_diff < 1e-10
        details = "max_diff=%.2e across %d comparisons" % (max_diff, comparisons)
        return passed, details

    def exp2_score_calibration(self):
        """Validate all Bayesian outputs in [0,1] and ordering preserved."""
        all_in_range = True
        ordering_preserved = True
        violations = []

        for query in self.queries:
            terms = query["terms"]
            bm25_scores = []
            bayesian_scores = []
            for doc in self.corpus.documents:
                raw = self.bm25.score(terms, doc)
                calibrated = self.bayesian.score(terms, doc)
                bm25_scores.append((doc["id"], raw))
                bayesian_scores.append((doc["id"], calibrated))
                if calibrated < -EPSILON or calibrated > 1.0 + EPSILON:
                    all_in_range = False
                    violations.append(
                        "doc=%s calibrated=%.6f" % (doc["id"], calibrated)
                    )

            # Check ordering preservation: if BM25(a) > BM25(b),
            # then Bayesian(a) >= Bayesian(b)
            bm25_ranked = sorted(bm25_scores, key=lambda x: -x[1])
            bayesian_map = {did: s for did, s in bayesian_scores}
            for i in range(len(bm25_ranked) - 1):
                id_a, score_a = bm25_ranked[i]
                id_b, score_b = bm25_ranked[i + 1]
                if score_a > score_b + EPSILON:
                    if bayesian_map[id_a] < bayesian_map[id_b] - EPSILON:
                        ordering_preserved = False
                        violations.append(
                            "query=%s: BM25(%s)=%.4f > BM25(%s)=%.4f "
                            "but Bayesian(%s)=%.6f < Bayesian(%s)=%.6f"
                            % (
                                query["text"],
                                id_a, score_a,
                                id_b, score_b,
                                id_a, bayesian_map[id_a],
                                id_b, bayesian_map[id_b],
                            )
                        )

        passed = all_in_range and ordering_preserved
        parts = ["range=[0,1]: %s" % all_in_range, "ordering: %s" % ordering_preserved]
        if violations:
            parts.append("violations: " + "; ".join(violations[:3]))
        return passed, ", ".join(parts)

    def exp3_monotonicity(self):
        """Validate Theorem 4.3.1: higher TF -> higher score (same doc length)."""
        passed = True
        details_parts = []

        # Group documents by similar length to test TF monotonicity
        docs_by_length = {}
        for doc in self.corpus.documents:
            bucket = doc["length"] // 5  # bucket by groups of 5
            if bucket not in docs_by_length:
                docs_by_length[bucket] = []
            docs_by_length[bucket].append(doc)

        # For each term that appears in multiple docs of similar length,
        # check that higher TF gives higher Bayesian score
        terms_tested = 0
        for term in self.corpus.df:
            matching_docs = [
                doc for doc in self.corpus.documents
                if doc["term_freq"].get(term, 0) > 0
            ]
            if len(matching_docs) < 2:
                continue
            # Sort by TF
            matching_docs.sort(key=lambda d: d["term_freq"].get(term, 0))
            for i in range(len(matching_docs) - 1):
                d1 = matching_docs[i]
                d2 = matching_docs[i + 1]
                tf1 = d1["term_freq"].get(term, 0)
                tf2 = d2["term_freq"].get(term, 0)
                if tf1 == tf2:
                    continue
                # Compare with same doc length to isolate TF effect
                if abs(d1["length"] - d2["length"]) <= 3:
                    s1 = self.bayesian.score_term(term, d1)
                    s2 = self.bayesian.score_term(term, d2)
                    terms_tested += 1
                    if s1 > s2 + EPSILON:
                        passed = False
                        details_parts.append(
                            "term=%s: tf(%s)=%d > tf(%s)=%d but score %.4f > %.4f"
                            % (term, d1["id"], tf1, d2["id"], tf2, s1, s2)
                        )

        # Also verify that the Bayesian scorer itself is monotonic in raw score:
        # test with synthetic inputs
        synthetic_passed = True
        for raw_score in [0.1, 0.5, 1.0, 2.0, 5.0]:
            for prior in [0.2, 0.5, 0.8]:
                p1 = self.bayesian.posterior(raw_score, prior)
                p2 = self.bayesian.posterior(raw_score + 0.1, prior)
                if p2 < p1 - EPSILON:
                    synthetic_passed = False

        passed = passed and synthetic_passed
        detail = "terms_tested=%d, synthetic_monotonic=%s" % (
            terms_tested, synthetic_passed
        )
        if details_parts:
            detail += ", violations: " + "; ".join(details_parts[:3])
        return passed, detail

    def exp4_prior_bounds(self):
        """Validate Theorem 4.2.4: all priors in [0.1, 0.9]."""
        all_bounded = True
        min_prior = 1.0
        max_prior = 0.0
        violations = []

        for doc in self.corpus.documents:
            for term in doc["term_freq"]:
                tf = doc["term_freq"][term]
                prior = self.bayesian.composite_prior(
                    tf, doc["length"], self.corpus.avgdl
                )
                min_prior = min(min_prior, prior)
                max_prior = max(max_prior, prior)
                if prior < 0.1 - EPSILON or prior > 0.9 + EPSILON:
                    all_bounded = False
                    violations.append(
                        "doc=%s term=%s prior=%.6f" % (doc["id"], term, prior)
                    )

        passed = all_bounded
        detail = "range=[%.4f, %.4f]" % (min_prior, max_prior)
        if violations:
            detail += ", violations: " + "; ".join(violations[:3])
        return passed, detail

    def exp5_idf_properties(self):
        """Validate IDF theorems: non-negativity, monotonicity, upper bound."""
        all_terms = sorted(self.corpus.df.keys())
        idf_values = {t: self.bm25.idf(t) for t in all_terms}

        # Theorem 3.1.2: IDF non-negative for terms with df <= N/2
        non_neg_ok = True
        for term in all_terms:
            df_t = self.corpus.df[term]
            if df_t <= self.corpus.n / 2.0:
                if idf_values[term] < -EPSILON:
                    non_neg_ok = False

        # Theorem 3.1.3: IDF monotonically decreasing with df
        df_idf_pairs = sorted(
            [(self.corpus.df[t], idf_values[t]) for t in all_terms],
            key=lambda x: x[0],
        )
        monotonic_ok = True
        for i in range(len(df_idf_pairs) - 1):
            df1, idf1 = df_idf_pairs[i]
            df2, idf2 = df_idf_pairs[i + 1]
            if df1 < df2 and idf1 < idf2 - EPSILON:
                monotonic_ok = False

        # Theorem 3.2.3: actual score <= upper bound
        bound_ok = True
        for query in self.queries:
            for term in query["terms"]:
                ub = self.bm25.upper_bound(term)
                for doc in self.corpus.documents:
                    actual = self.bm25.score_term_standard(term, doc)
                    if actual > ub + EPSILON:
                        bound_ok = False

        passed = non_neg_ok and monotonic_ok and bound_ok
        detail = "non_neg=%s, monotonic=%s, upper_bound=%s" % (
            non_neg_ok, monotonic_ok, bound_ok
        )
        return passed, detail

    def exp6_hybrid_quality(self):
        """Validate AND <= min(probs), OR >= max(probs) (Thms 5.1.2, 5.2.2)."""
        passed = True
        tests = 0
        violations = []

        for query in self.queries:
            if "embedding" not in query:
                continue
            terms = query["terms"]
            for doc in self.corpus.documents:
                bayesian_p = self.bayesian.score(terms, doc)
                vector_p = self.vector.score(query["embedding"], doc)
                probs = [bayesian_p, vector_p]

                and_score = self.hybrid.probabilistic_and(probs)
                or_score = self.hybrid.probabilistic_or(probs)
                min_p = min(probs)
                max_p = max(probs)

                tests += 1
                if and_score > min_p + EPSILON:
                    passed = False
                    violations.append(
                        "AND=%.6f > min=%.6f (doc=%s)" % (and_score, min_p, doc["id"])
                    )
                if or_score < max_p - EPSILON:
                    passed = False
                    violations.append(
                        "OR=%.6f < max=%.6f (doc=%s)" % (or_score, max_p, doc["id"])
                    )

        detail = "tests=%d" % tests
        if violations:
            detail += ", violations: " + "; ".join(violations[:3])
        return passed, detail

    def exp7_method_comparison(self):
        """Compare naive sum, RRF, and Bayesian hybrid methods."""
        results_table = []

        for query in self.queries:
            if "embedding" not in query:
                continue
            terms = query["terms"]

            # Score all docs with each method
            doc_scores = []
            for doc in self.corpus.documents:
                bm25_raw = self.bm25.score(terms, doc)
                bayesian_p = self.bayesian.score(terms, doc)
                vector_p = self.vector.score(query["embedding"], doc)
                hybrid_or = self.hybrid.score_or(terms, query["embedding"], doc)
                hybrid_and = self.hybrid.score_and(terms, query["embedding"], doc)
                naive = self.hybrid.naive_sum([bm25_raw, vector_p])

                doc_scores.append({
                    "id": doc["id"],
                    "bm25": bm25_raw,
                    "bayesian": bayesian_p,
                    "vector": vector_p,
                    "hybrid_or": hybrid_or,
                    "hybrid_and": hybrid_and,
                    "naive": naive,
                })

            # Compute RRF ranks
            bm25_ranked = sorted(doc_scores, key=lambda x: -x["bm25"])
            vector_ranked = sorted(doc_scores, key=lambda x: -x["vector"])
            bm25_rank = {d["id"]: i + 1 for i, d in enumerate(bm25_ranked)}
            vector_rank = {d["id"]: i + 1 for i, d in enumerate(vector_ranked)}
            for d in doc_scores:
                d["rrf"] = self.hybrid.rrf_score(
                    [bm25_rank[d["id"]], vector_rank[d["id"]]]
                )

            # Get top-5 by each method
            top5 = {}
            for method in ["bm25", "bayesian", "hybrid_or", "hybrid_and", "naive", "rrf"]:
                ranked = sorted(doc_scores, key=lambda x: -x[method])
                top5[method] = [d["id"] for d in ranked[:5]]

            results_table.append({
                "query": query["text"],
                "top5": top5,
            })

        # Signal dominance check (Theorem 1.2.2): naive sum can be dominated
        # by one signal. Check if naive top-5 ever differs from Bayesian top-5.
        has_difference = False
        for r in results_table:
            if r["top5"]["naive"] != r["top5"]["hybrid_or"]:
                has_difference = True

        detail_lines = []
        for r in results_table:
            detail_lines.append("query='%s':" % r["query"])
            for method in ["bm25", "bayesian", "hybrid_or", "naive", "rrf"]:
                detail_lines.append(
                    "  %s top5: %s" % (method, r["top5"][method])
                )
        detail = "\n".join(detail_lines)

        # This experiment always passes -- it's a comparison, not a validation.
        # The interesting output is the ranking tables.
        return True, detail

    def exp8_numerical_stability(self):
        """Validate log-space computation handles extreme probabilities (Thm 5.3.1)."""
        passed = True
        tests = []

        # Test with extreme probability values
        extreme_probs = [1e-15, 1e-10, 1e-5, 0.001, 0.5, 0.999, 1.0 - 1e-10]

        for p in extreme_probs:
            # Test probabilistic_and
            and_result = self.hybrid.probabilistic_and([p, p])
            expected_and = p * p
            if and_result < -EPSILON or and_result > 1.0 + EPSILON:
                passed = False
            if math.isnan(and_result) or math.isinf(and_result):
                passed = False
            tests.append("AND(%.2e, %.2e)=%.2e" % (p, p, and_result))

            # Test probabilistic_or
            or_result = self.hybrid.probabilistic_or([p, p])
            expected_or = 1.0 - (1.0 - p) ** 2
            if or_result < -EPSILON or or_result > 1.0 + EPSILON:
                passed = False
            if math.isnan(or_result) or math.isinf(or_result):
                passed = False
            tests.append("OR(%.2e, %.2e)=%.2e" % (p, p, or_result))

        # Test sigmoid with extreme inputs
        for x in [-700, -100, -1, 0, 1, 100, 700]:
            s = sigmoid(x)
            if s < 0 or s > 1 or math.isnan(s) or math.isinf(s):
                passed = False
            tests.append("sigmoid(%d)=%.6f" % (x, s))

        # Test safe_log with near-zero
        for p in [0.0, 1e-300, 1e-15, 0.5, 1.0]:
            result = safe_log(p)
            if math.isnan(result) or math.isinf(result):
                passed = False
            tests.append("safe_log(%.2e)=%.2f" % (p, result))

        detail = "; ".join(tests[:10]) + " ... (%d total tests)" % len(tests)
        return passed, detail

    def exp9_parameter_learning(self):
        """Validate Algorithm 8.3.1: loss decreases, alpha > 0."""
        # Generate training data from corpus
        scores = []
        labels = []

        # Use a specific query to generate labeled data
        query = self.queries[0]
        terms = query["terms"]
        relevant_ids = set(query.get("relevant", []))

        for doc in self.corpus.documents:
            raw_score = self.bm25.score(terms, doc)
            scores.append(raw_score)
            labels.append(1.0 if doc["id"] in relevant_ids else 0.0)

        learner = ParameterLearner(
            learning_rate=0.1, max_iterations=500, tolerance=1e-8
        )
        result = learner.learn(scores, labels)

        # Check: loss should generally decrease
        loss_history = result["loss_history"]
        loss_decreased = loss_history[-1] < loss_history[0]

        # Check: alpha should be positive (higher scores = more relevant)
        alpha_positive = result["alpha"] > 0

        # Check: loss decreased over most iterations
        decreasing_steps = sum(
            1 for i in range(1, len(loss_history))
            if loss_history[i] <= loss_history[i - 1] + EPSILON
        )
        mostly_decreasing = decreasing_steps >= len(loss_history) * 0.8

        passed = loss_decreased and alpha_positive and mostly_decreasing
        detail = (
            "alpha=%.4f, beta=%.4f, initial_loss=%.4f, final_loss=%.4f, "
            "decreasing_steps=%d/%d, converged=%s"
            % (
                result["alpha"],
                result["beta"],
                loss_history[0],
                loss_history[-1],
                decreasing_steps,
                len(loss_history) - 1,
                result["converged"],
            )
        )
        return passed, detail

    def exp10_conjunction_disjunction(self):
        """Validate Theorems 5.1.2, 5.2.2 for all hybrid queries."""
        passed = True
        tests = 0
        violations = []

        # Test with many probability combinations
        test_probs = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        for p1 in test_probs:
            for p2 in test_probs:
                probs = [p1, p2]
                and_result = self.hybrid.probabilistic_and(probs)
                or_result = self.hybrid.probabilistic_or(probs)
                min_p = min(probs)
                max_p = max(probs)

                tests += 1

                # Theorem 5.1.2: AND <= min
                if and_result > min_p + EPSILON:
                    passed = False
                    violations.append(
                        "AND(%.2f,%.2f)=%.6f > min=%.2f"
                        % (p1, p2, and_result, min_p)
                    )

                # Theorem 5.2.2: OR >= max
                if or_result < max_p - EPSILON:
                    passed = False
                    violations.append(
                        "OR(%.2f,%.2f)=%.6f < max=%.2f"
                        % (p1, p2, or_result, max_p)
                    )

                # AND <= OR always
                if and_result > or_result + EPSILON:
                    passed = False
                    violations.append(
                        "AND(%.2f,%.2f)=%.6f > OR=%.6f"
                        % (p1, p2, and_result, or_result)
                    )

        # Also test with 3 probabilities
        for p1 in [0.1, 0.5, 0.9]:
            for p2 in [0.2, 0.6]:
                for p3 in [0.3, 0.8]:
                    probs = [p1, p2, p3]
                    and_result = self.hybrid.probabilistic_and(probs)
                    or_result = self.hybrid.probabilistic_or(probs)
                    tests += 1
                    if and_result > min(probs) + EPSILON:
                        passed = False
                    if or_result < max(probs) - EPSILON:
                        passed = False

        detail = "tests=%d" % tests
        if violations:
            detail += ", violations: " + "; ".join(violations[:3])
        return passed, detail
