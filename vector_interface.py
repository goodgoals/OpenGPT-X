"""Opaque interface to externally managed token vectors.

This module never initializes, serializes, or persists vectors.
It only forwards calls to an external backend object.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


_backend: Any = None


def register_backend(backend: Any) -> None:
    """Register an external vector backend.

    The backend must provide:
    - get_vector(token_id) -> np.ndarray
    - update_vectors(token_ids, delta) -> None
    """
    global _backend
    _backend = backend


def get_vector(token_id: int) -> np.ndarray:
    if _backend is None:
        raise RuntimeError("No vector backend registered")
    return _backend.get_vector(token_id)


def update_vectors(token_ids: Iterable[int], delta: float) -> None:
    if _backend is None:
        raise RuntimeError("No vector backend registered")
    _backend.update_vectors(list(token_ids), float(delta))
