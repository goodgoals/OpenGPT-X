"""End-to-end demo runner for structural-pressure language learning."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import survival_filter
import trainer
import vector_interface


TOKENS_FILE = Path("tokens.csv")


@dataclass(frozen=True)
class Token:
    token_id: int
    token_string: str
    token_type: str


class DemoExternalVectorBackend:
    """Demo stand-in for an external vector service.

    The model code only sees backend function calls via vector_interface.
    No serialization or file persistence is implemented.
    """

    def __init__(self, token_ids: Sequence[int], dim: int = 10, seed: int = 11) -> None:
        rng = random.Random(seed)
        self._vectors: Dict[int, List[float]] = {
            token_id: [rng.uniform(-1.0, 1.0) for _ in range(dim)] for token_id in token_ids
        }

    def get_vector(self, token_id: int) -> Sequence[float]:
        return self._vectors[token_id]

    def update_vectors(self, token_ids: Sequence[int], delta: float) -> None:
        for left, right in zip(token_ids, token_ids[1:]):
            left_vector = self._vectors[left]
            right_vector = self._vectors[right]
            for i, (lv, rv) in enumerate(zip(left_vector, right_vector)):
                step = (rv - lv) * delta
                left_vector[i] = lv + step
                right_vector[i] = rv - step


def load_tokens(path: Path) -> List[Token]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            Token(
                token_id=int(row["token_id"]),
                token_string=row["token_string"],
                token_type=row["token_type"],
            )
            for row in rows
        ]


def decode(sequence: Sequence[int], vocab: Dict[int, str]) -> str:
    return " ".join(vocab[token_id] for token_id in sequence)


def run_generations(token_ids: List[int], vocab: Dict[int, str], count: int) -> None:
    for index in range(count):
        sequence = trainer.generate_sequence(token_ids)
        survived, score = survival_filter.survives(sequence)
        print(f"sample {index + 1}: {decode(sequence, vocab)}")
        print(f"  coherence={score:.4f} survive={survived}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Structural-pressure LM demo")
    parser.add_argument("--input", default="Jake started the car.")
    parser.add_argument("--before", type=int, default=3)
    parser.add_argument("--after", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=250)
    args = parser.parse_args()

    tokens = load_tokens(TOKENS_FILE)
    token_ids = [token.token_id for token in tokens]
    vocab = {token.token_id: token.token_string for token in tokens}

    vector_interface.register_backend(DemoExternalVectorBackend(token_ids))

    print(f"Input: {args.input}")
    print("Expect poor output before training.")

    print("\n--- Before training ---")
    run_generations(token_ids, vocab, args.before)

    training_stats = trainer.train(token_ids, iterations=args.iterations)
    print("\nTraining stats:")
    for key, value in training_stats.items():
        print(f"  {key}: {value}")

    print("\n--- After training ---")
    run_generations(token_ids, vocab, args.after)


if __name__ == "__main__":
    main()
