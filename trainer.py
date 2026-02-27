"""Training loop driven only by survival pressure.

Prediction, filtering, and vector updates are intentionally separated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import probability_logic
import survival_filter
import vector_interface


BOS_ID = 0
EOS_ID = 1
TokenId = int


@dataclass
class TrainingStep:
    sequence: List[TokenId]
    survived: bool
    score: float


def generate_sequence(token_ids: Sequence[TokenId], max_len: int = 8) -> List[TokenId]:
    """Generate one sequence with no learning signal involved."""
    sequence = [BOS_ID]

    for _ in range(max_len):
        next_token = probability_logic.choose_next_token(token_ids, sequence)
        sequence.append(next_token)
        if next_token == EOS_ID:
            break

    if sequence[-1] != EOS_ID:
        sequence.append(EOS_ID)

    return sequence


def apply_survival_update(
    sequence: Sequence[TokenId],
    survived: bool,
    reinforce_delta: float,
    weaken_delta: float,
) -> None:
    """Apply local vector update hooks after survive/die decision."""
    delta = reinforce_delta if survived else weaken_delta
    vector_interface.update_vectors(sequence, delta)


def train_step(
    token_ids: Sequence[TokenId],
    survive_threshold: float = 0.05,
    reinforce_delta: float = 0.02,
    weaken_delta: float = -0.005,
) -> TrainingStep:
    sequence = generate_sequence(token_ids)
    survived, score = survival_filter.survives(sequence, threshold=survive_threshold)
    apply_survival_update(
        sequence,
        survived=survived,
        reinforce_delta=reinforce_delta,
        weaken_delta=weaken_delta,
    )
    return TrainingStep(sequence=list(sequence), survived=survived, score=score)


def train(
    token_ids: Sequence[TokenId],
    iterations: int = 200,
    survive_threshold: float = 0.05,
    reinforce_delta: float = 0.02,
    weaken_delta: float = -0.005,
) -> Dict[str, float]:
    """Run local updates only; no global objective is optimized."""
    history = [
        train_step(
            token_ids,
            survive_threshold=survive_threshold,
            reinforce_delta=reinforce_delta,
            weaken_delta=weaken_delta,
        )
        for _ in range(iterations)
    ]

    survivors = sum(1 for step in history if step.survived)
    avg_score = sum(step.score for step in history) / len(history) if history else 0.0

    return {
        "iterations": float(iterations),
        "survivors": float(survivors),
        "survive_rate": survivors / len(history) if history else 0.0,
        "avg_score": avg_score,
    }
