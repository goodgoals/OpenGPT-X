"""Sequence survival gate using only vector coherence."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import vector_interface


TokenId = int


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vector dimensions must match")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def coherence_score(sequence: Sequence[TokenId]) -> float:
    if len(sequence) < 2:
        return 0.0

    similarities = []
    for left, right in zip(sequence, sequence[1:]):
        vector_left = vector_interface.get_vector(left)
        vector_right = vector_interface.get_vector(right)
        similarities.append(_cosine_similarity(vector_left, vector_right))

    return sum(similarities) / len(similarities) if similarities else 0.0


def survives(sequence: Sequence[TokenId], threshold: float = 0.05) -> Tuple[bool, float]:
    score = coherence_score(sequence)
    return score >= threshold, score
