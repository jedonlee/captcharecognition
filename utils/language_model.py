import logging
import numpy as np
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class CharNGramLM:
    """
    Character-level n-gram language model

    Used in CTC Beam Search decoding to introduce language constraints,
    applying penalties/rewards for character transition probabilities
    on top of visual probabilities.

    Usage:
        lm = CharNGramLM(order=2, smoothing=0.01)
        lm.build(captcha_texts)
        score = lm.log_prob('A', ['B'])  # P(A|B)
    """

    def __init__(self, order: int = 2, smoothing: float = 0.01):
        self.order = order
        self.smoothing = smoothing
        self._counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self._context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self._vocab: set = set()
        self._built = False

    def build(self, texts: List[str]):
        """
        Build n-gram statistics from text list

        Args:
            texts: list of CAPTCHA texts (e.g. ["AB12C", "X9Y3K"])
        """
        for text in texts:
            if not text:
                continue
            padded = ('<s>',) * (self.order - 1) + tuple(text) + ('</s>',)
            for i in range(len(padded) - self.order + 1):
                context = padded[i:i + self.order - 1]
                target = padded[i + self.order - 1]
                self._counts[context][target] += 1
                self._context_counts[context] += 1
                self._vocab.add(target)

        self._vocab.add('<s>')
        self._vocab.add('</s>')
        self._built = True
        logger.info(f"n-gram LM built: order={self.order}, vocab_size={len(self._vocab)}, "
                    f"unique_contexts={len(self._counts)}")

    def log_prob(self, char: str, context: List[str]) -> float:
        if not self._built:
            return 0.0

        ctx = tuple(context[-(self.order - 1):]) if context else ('<s>',) * (self.order - 1)
        if len(ctx) < self.order - 1:
            ctx = ('<s>',) * (self.order - 1 - len(ctx)) + ctx

        count_context = self._context_counts.get(ctx, 0)
        count_char = self._counts.get(ctx, {}).get(char, 0)

        vocab_size = len(self._vocab)
        prob = (count_char + self.smoothing) / (count_context + self.smoothing * vocab_size)

        return max(np.log(prob), -20.0)

    def sequence_log_prob(self, text: str) -> float:
        if not text:
            return 0.0
        score = 0.0
        context: List[str] = []
        for ch in text:
            score += self.log_prob(ch, context)
            context.append(ch)
        return score / max(len(text), 1)

    def get_transition_prob(self, prev_char: str, next_char: str) -> float:
        return np.exp(self.log_prob(next_char, [prev_char]))

    def save(self, path: str):
        import pickle
        data = {
            'order': self.order,
            'smoothing': self.smoothing,
            'counts': {str(k): dict(v) for k, v in self._counts.items()},
            'context_counts': {str(k): v for k, v in self._context_counts.items()},
            'vocab': list(self._vocab),
            'built': self._built,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"n-gram LM saved: {path}")

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.order = data['order']
        self.smoothing = data['smoothing']
        self._counts = defaultdict(Counter, {eval(k): Counter(v) for k, v in data['counts'].items()})
        self._context_counts = defaultdict(int, {eval(k): v for k, v in data['context_counts'].items()})
        self._vocab = set(data['vocab'])
        self._built = data['built']
        logger.info(f"n-gram LM loaded: {path}")