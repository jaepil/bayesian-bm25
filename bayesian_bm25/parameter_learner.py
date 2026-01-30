"""Gradient descent parameter learner for Bayesian BM25 calibration."""

import math

from bayesian_bm25.math_utils import sigmoid, EPSILON


class ParameterLearner:
    """Learns optimal alpha and beta parameters via gradient descent.

    Minimizes cross-entropy loss (Definition 8.1.1) between predicted
    relevance probabilities and binary labels using Algorithm 8.3.1.
    """

    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def cross_entropy_loss(self, scores, labels, alpha, beta):
        """Binary cross-entropy loss (Definition 8.1.1).

        L = -1/N * sum(y * log(p) + (1-y) * log(1-p))
        where p = sigmoid(alpha * (s - beta)).
        """
        n = len(scores)
        total_loss = 0.0
        for s, y in zip(scores, labels):
            p = sigmoid(alpha * (s - beta))
            p = max(EPSILON, min(1.0 - EPSILON, p))
            total_loss -= y * math.log(p) + (1.0 - y) * math.log(1.0 - p)
        return total_loss / n

    def learn(self, scores, labels):
        """Learn alpha and beta by gradient descent (Algorithm 8.3.1).

        Args:
            scores: List of raw BM25 scores.
            labels: List of binary relevance labels (0 or 1).

        Returns:
            Dict with keys: alpha, beta, loss_history, converged.
        """
        alpha = 1.0
        beta = 0.0
        n = len(scores)
        loss_history = []

        for iteration in range(self.max_iterations):
            loss = self.cross_entropy_loss(scores, labels, alpha, beta)
            loss_history.append(loss)

            if iteration > 0 and abs(loss_history[-2] - loss) < self.tolerance:
                return {
                    "alpha": alpha,
                    "beta": beta,
                    "loss_history": loss_history,
                    "converged": True,
                }

            # Compute gradients
            grad_alpha = 0.0
            grad_beta = 0.0
            for s, y in zip(scores, labels):
                p = sigmoid(alpha * (s - beta))
                p = max(EPSILON, min(1.0 - EPSILON, p))
                error = p - y
                grad_alpha += error * (s - beta)
                grad_beta += error * (-alpha)
            grad_alpha /= n
            grad_beta /= n

            alpha -= self.learning_rate * grad_alpha
            beta -= self.learning_rate * grad_beta

        final_loss = self.cross_entropy_loss(scores, labels, alpha, beta)
        loss_history.append(final_loss)

        return {
            "alpha": alpha,
            "beta": beta,
            "loss_history": loss_history,
            "converged": False,
        }
