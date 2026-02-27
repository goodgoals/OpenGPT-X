"""Sequence survival gate using only vector coherence."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

import vector_interface


TokenId = int


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def coherence_score(sequence: Sequence[TokenId]) -> float:
    if len(sequence) < 2:
        return 0.0

    sims = []
    for left, right in zip(sequence, sequence[1:]):
        vec_l = vector_interface.get_vector(left)
        vec_r = vector_interface.get_vector(right)
        sims.append(_cosine_similarity(vec_l, vec_r))

    return float(np.mean(sims)) if sims else 0.0


def survives(sequence: Sequence[TokenId], threshold: float = 0.05) -> Tuple[bool, float]:
    score = coherence_score(sequence)
    return score >= threshold, score
