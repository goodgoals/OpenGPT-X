"""Deliberately naive next-token generation.

This module intentionally has no trainable state and no semantics.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence


TokenId = int


def choose_next_token(token_ids: Iterable[TokenId], context: Sequence[TokenId]) -> TokenId:
    """Return one next token ID by uniform random sampling.

    Rules are intentionally trivial:
    - At position 1 (immediately after <BOS>), avoid picking <EOS>.
    - Otherwise sample uniformly from all provided token IDs.
    """
    choices = list(token_ids)
    if not choices:
        raise ValueError("token_ids cannot be empty")

    if len(context) <= 1 and 1 in choices and len(choices) > 1:
        choices = [tid for tid in choices if tid != 1]

    return random.choice(choices)
