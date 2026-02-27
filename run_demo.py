"""End-to-end demonstration of structural-pressure learning."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

import survival_filter
import trainer
import vector_interface


TOKENS_FILE = Path("tokens.csv")


@dataclass
class Token:
    token_id: int
    token_string: str
    token_type: str


class DemoExternalVectorBackend:
    """Demo-only stand-in for an external vector service.

    This behaves like a black-box provider through function calls.
    It does not serialize vectors.
    """

    def __init__(self, token_ids: List[int], dim: int = 12, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self._vectors: Dict[int, np.ndarray] = {
            tid: rng.normal(0.0, 1.0, size=(dim,)).astype(np.float32) for tid in token_ids
        }

    def get_vector(self, token_id: int) -> np.ndarray:
        return self._vectors[token_id]

    def update_vectors(self, token_ids: List[int], delta: float) -> None:
        # Local, pair-based adjustment only.
        for left, right in zip(token_ids, token_ids[1:]):
            v_l = self._vectors[left]
            v_r = self._vectors[right]
            adjustment = (v_r - v_l) * delta
            self._vectors[left] = v_l + adjustment
            self._vectors[right] = v_r - adjustment


def load_tokens(path: Path) -> List[Token]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        return [
            Token(
                token_id=int(row["token_id"]),
                token_string=row["token_string"],
                token_type=row["token_type"],
            )
            for row in rows
        ]


def decode(sequence: List[int], vocab: Dict[int, str]) -> str:
    return " ".join(vocab[tid] for tid in sequence)


def main() -> None:
    prompt = "Jake started the car."
    print(f"Input: {prompt}")

    tokens = load_tokens(TOKENS_FILE)
    token_ids = [t.token_id for t in tokens]
    vocab = {t.token_id: t.token_string for t in tokens}

    backend = DemoExternalVectorBackend(token_ids)
    vector_interface.register_backend(backend)

    print("\n--- Before training ---")
    for _ in range(3):
        seq = trainer.generate_sequence(token_ids)
        alive, score = survival_filter.survives(seq)
        print(f"generated: {decode(seq, vocab)}")
        print(f"coherence: {score:.4f} | survive: {alive}")

    stats = trainer.train(token_ids, iterations=300, survive_threshold=0.05)
    print("\nTraining stats:", stats)

    print("\n--- After training ---")
    for _ in range(5):
        seq = trainer.generate_sequence(token_ids)
        alive, score = survival_filter.survives(seq)
        print(f"generated: {decode(seq, vocab)}")
        print(f"coherence: {score:.4f} | survive: {alive}")


if __name__ == "__main__":
    main()
