"""Character-trigram language model over real package names, add-one smoothed (BUILD_SPEC §7).

``mean_logprob(name)`` is the mean per-character natural-log probability, including
the end-of-name symbol. Names that "look like" real package names score higher.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable

BOS, EOS = "\x02", "\x03"


class CharTrigramLM:
    """P(c | two previous chars) with add-one (Laplace) smoothing."""

    def __init__(self, names: Iterable[str]) -> None:
        names = list(names)
        self.trigrams: Counter[str] = Counter()
        self.bigrams: Counter[str] = Counter()
        vocab = {EOS}
        for name in names:
            padded = BOS + BOS + name + EOS
            vocab.update(name)
            for i in range(2, len(padded)):
                self.trigrams[padded[i - 2 : i + 1]] += 1
                self.bigrams[padded[i - 2 : i]] += 1
        self.vocab_size = len(vocab) + 1  # +1 for unseen characters
        scores = [self.mean_logprob(n) for n in names]
        self.mean = statistics.fmean(scores)
        self.std = statistics.pstdev(scores) or 1.0

    def mean_logprob(self, name: str) -> float:
        """Mean log P per character (end symbol included)."""
        padded = BOS + BOS + name + EOS
        total = 0.0
        for i in range(2, len(padded)):
            tri, ctx = padded[i - 2 : i + 1], padded[i - 2 : i]
            total += math.log((self.trigrams[tri] + 1) / (self.bigrams[ctx] + self.vocab_size))
        return total / (len(padded) - 2)

    def zscore(self, name: str) -> float:
        """``mean_logprob`` standardized against the training names."""
        return (self.mean_logprob(name) - self.mean) / self.std
