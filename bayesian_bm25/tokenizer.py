"""Simple tokenizer for BM25 text processing."""

import re


class Tokenizer:
    """Lowercase and split on non-alphanumeric characters."""

    _SPLIT_PATTERN = re.compile(r"[^a-z0-9]+")

    def tokenize(self, text):
        """Return list of lowercase alphanumeric tokens."""
        lowered = text.lower()
        tokens = self._SPLIT_PATTERN.split(lowered)
        return [t for t in tokens if t]
