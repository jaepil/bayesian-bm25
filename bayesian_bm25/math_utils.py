"""Standalone math utility functions for Bayesian BM25."""

import math

EPSILON = 1e-10


def sigmoid(x):
    """Numerically stable sigmoid function (Definition 2.1.2).

    Uses the identity: sigmoid(x) = exp(x)/(1+exp(x)) for x >= 0
    and 1/(1+exp(-x)) for x < 0 to avoid overflow.
    """
    if x >= 0:
        ez = math.exp(-x)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(x)
        return ez / (1.0 + ez)


def safe_log(p):
    """Logarithm with EPSILON clamping to avoid log(0) (Definition 5.3.2)."""
    return math.log(max(p, EPSILON))


def safe_prob(p):
    """Clamp probability to [EPSILON, 1-EPSILON]."""
    return clamp(p, EPSILON, 1.0 - EPSILON)


def clamp(value, low, high):
    """Clamp a value to [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def dot_product(a, b):
    """Dot product of two vectors."""
    return sum(ai * bi for ai, bi in zip(a, b))


def vector_magnitude(v):
    """Euclidean magnitude of a vector."""
    return math.sqrt(sum(vi * vi for vi in v))


def cosine_similarity(a, b):
    """Cosine similarity between two vectors. Returns 0.0 for zero vectors."""
    mag_a = vector_magnitude(a)
    mag_b = vector_magnitude(b)
    if mag_a < EPSILON or mag_b < EPSILON:
        return 0.0
    return dot_product(a, b) / (mag_a * mag_b)
