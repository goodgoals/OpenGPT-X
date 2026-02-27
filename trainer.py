"""Training loop driven by survival pressure only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import probability_logic
import survival_filter
import vector_interface


TokenId = int


@dataclass
class TrainingStep:
    sequence: List[TokenId]
    survived: bool
    score: float


def generate_sequence(token_ids: Sequence[TokenId], max_len: int = 8) -> List[TokenId]:
    sequence = [0]  # <BOS>
    for _ in range(max_len):
        nxt = probability_logic.choose_next_token(token_ids, sequence)
        sequence.append(nxt)
        if nxt == 1:  # <EOS>
            break
    if sequence[-1] != 1:
        sequence.append(1)
    return sequence


def train_step(
    token_ids: Sequence[TokenId],
    survive_threshold: float = 0.05,
    reinforce_delta: float = 0.03,
    weaken_delta: float = -0.01,
) -> TrainingStep:
    seq = generate_sequence(token_ids)
    alive, score = survival_filter.survives(seq, threshold=survive_threshold)

    if alive:
        vector_interface.update_vectors(seq, reinforce_delta)
    else:
        vector_interface.update_vectors(seq, weaken_delta)

    return TrainingStep(sequence=seq, survived=alive, score=score)


def train(
    token_ids: Sequence[TokenId],
    iterations: int = 200,
    survive_threshold: float = 0.05,
) -> Dict[str, float]:
    outcomes = [train_step(token_ids, survive_threshold=survive_threshold) for _ in range(iterations)]
    survive_rate = sum(step.survived for step in outcomes) / max(len(outcomes), 1)
    avg_score = sum(step.score for step in outcomes) / max(len(outcomes), 1)
    return {
        "iterations": float(iterations),
        "survive_rate": survive_rate,
        "avg_score": avg_score,
    }
