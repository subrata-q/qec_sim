"""Logical operator basis computation and logical failure detection.

Determines, from the code's check matrices, a basis of representative
logical operators (undetectable, non-trivial Pauli operators), then lets
`run_experiment` decide whether a given fault residual (true error XOR
decoder correction) anticommutes with / overlaps one of them, i.e. whether
the shot is a logical failure.
"""

from __future__ import annotations
import numpy as np
from .gf2 import nullspace, quotient_basis
from .spacetime_pcm import build_spatial_matrix


def compute_logical_operators(
    H_X: np.ndarray, H_Z: np.ndarray | None = None, H_Y: np.ndarray | None = None
) -> dict:
    """Compute a logical operator basis, dispatching on which mode applies.

    Mode is inferred from which check matrices are supplied (see package
    docstring): "single" (`H_X` only), "css" (`H_X` + `H_Z`), or "xyz"
    (`H_X` + `H_Z` + `H_Y`).

    - single: logical operators are simply ker(H_X); any fault pattern
      that produces no syndrome is undetectable.
    - css: X-type and Z-type errors are tracked independently. Logical Z
      operators are ker(H_X) modulo the Z-stabilizer rowspace (`H_Z`), and
      vice versa for logical X — see `gf2.quotient_basis`.
    - xyz: X/Z faults are correlated (as in an XYZ/Y-biased code), so
      logical operators are computed jointly over the concatenated
      [X-part | Z-part] spatial matrix, quotiented against its own
      X/Z-swapped version (a logical operator must anticommute with the
      "other" check type through the swap).

    Args:
        H_X: X-type (or single-channel) check matrix, shape (checks, n).
        H_Z: Z-type check matrix, required for "css"/"xyz" modes.
        H_Y: Y-type check matrix; supplying it selects "xyz" mode.

    Returns:
        Dict with key "mode" plus mode-specific logical operator bases:
            single: {"mode": "single", "logical": <basis>}
            css: {"mode": "css", "logical_z": <basis>, "logical_x": <basis>}
            xyz: {"mode": "xyz", "logical": <basis>}
        Each basis is a uint8 array of shape (k, n), one operator per row.
    """
    H_X = np.asarray(H_X, dtype=np.uint8)
    n = H_X.shape[1]

    if H_Z is None and H_Y is None:
        return {"mode": "single", "logical": nullspace(H_X)}

    H_Z = np.asarray(H_Z, dtype=np.uint8)

    if H_Y is None:
        logical_z = quotient_basis(H_X, H_Z)
        logical_x = quotient_basis(H_Z, H_X)
        return {"mode": "css", "logical_z": logical_z, "logical_x": logical_x}

    H_Y = np.asarray(H_Y, dtype=np.uint8)
    H_spatial = build_spatial_matrix(H_X, H_Z, H_Y)
    swapped = np.hstack([H_spatial[:, n:], H_spatial[:, :n]])
    logical = quotient_basis(H_spatial, swapped)
    return {"mode": "xyz", "logical": logical}


def logical_failure(residual: np.ndarray, logical_basis: np.ndarray) -> bool:
    """Check whether a fault residual overlaps any logical operator.

    A shot fails logically iff the residual (true fault XOR decoder
    correction, restricted to a channel) has nonzero GF(2) inner product
    with at least one basis vector — i.e. it is not orthogonal to the
    full logical operator subspace.

    Args:
        residual: 1D binary residual vector for one channel.
        logical_basis: Logical operator basis for that channel, as
            returned by `compute_logical_operators` (shape (k, n)).

    Returns:
        True if the residual triggers a logical failure; False if there
        are no logical operators (`logical_basis` is empty) or the
        residual is orthogonal to all of them.
    """
    if logical_basis.shape[0] == 0:
        return False
    return bool(((logical_basis.astype(np.uint8) @ residual.astype(np.uint8)) % 2).any())
