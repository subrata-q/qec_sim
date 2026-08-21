"""Space-time parity-check matrix (PCM) generation and file IO.

The space-time PCM unrolls a single "spatial" (single-round) check matrix
over `r` rounds of measurement, and adds a temporal difference so that
each round's checks report the same syndrome bit twice in a row unless
something changed — the standard construction for decoding repeated
stabilizer measurements with a matching/BP-style decoder.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, kron, eye, hstack, save_npz, load_npz, issparse


def build_delta_r(r: int) -> np.ndarray:
    """Build the `r x r` lower-bidiagonal temporal difference matrix.

    Identity on the diagonal plus a subdiagonal of 1s, so that applying it
    to a sequence of per-round syndromes yields, for round `i > 0`, the
    XOR of round `i` and round `i-1` (and just round 0 unchanged for
    `i == 0`) — i.e. only *changes* in the syndrome across consecutive
    rounds are reported, which is what makes repeated-measurement
    detectors sparse/local in time.

    Args:
        r: Number of rounds.

    Returns:
        `(r, r)` uint8 array.
    """
    delta = np.eye(r, dtype=np.uint8)
    for i in range(1, r):
        delta[i, i - 1] = 1
    return delta


def meas_rounds(
    r: int, perfect_first_round: bool = False, perfect_last_round: bool = False
) -> int:
    """Number of rounds that carry measurement noise.

    Each idealized round drops its own block of `m_total` measurement-flip
    columns from the space-time PCM, so this is the one number that
    `generate_space_time_pcm`, `error_model.sample_shot` and
    `error_model.build_priors` must all agree on. They call it rather than
    recomputing it, since a disagreement shows up only as a column-count
    mismatch deep inside the decoder.

    Args:
        r: Number of rounds.
        perfect_first_round: Whether round 0's measurements are noiseless.
        perfect_last_round: Whether round `r-1`'s measurements are noiseless.

    Returns:
        `r` minus the number of idealized rounds.

    Raises:
        ValueError: If `r < 1`, or if idealizing leaves no noisy round --
            `r=1` with either flag, or `r=2` with both. There would be no
            measurement columns at all, which is a silently wrong model
            rather than a useful one.
    """
    if r < 1:
        raise ValueError("r must be at least 1.")
    idealized = int(bool(perfect_first_round)) + int(bool(perfect_last_round))
    if idealized >= r:
        raise ValueError(
            f"r={r} with perfect_first_round={bool(perfect_first_round)} and "
            f"perfect_last_round={bool(perfect_last_round)} leaves no noisy "
            "round; at least one round must carry measurement noise."
        )
    return r - idealized


def build_spatial_matrix(
    H_X: np.ndarray, H_Z: np.ndarray | None = None, H_Y: np.ndarray | None = None
) -> np.ndarray:
    """Build the single-round check matrix, dispatching on which mode applies.

    - single (`H_Z`, `H_Y` both `None`): returns `H_X` unchanged.
    - css (`H_Z` given, `H_Y` `None`): block-diagonal stack of `H_X` and
      `H_Z`, each acting on its own half of a `2n`-wide column space
      (`[H_X 0; 0 H_Z]`), so X-type and Z-type checks are independent.
    - xyz (`H_X`, `H_Z`, `H_Y` all given): as css, plus a `H_Y` block
      whose *same* `n` columns are duplicated into both the X-half and
      Z-half (`[H_Y H_Y]`), since a Y check must fire on both an X-type
      and Z-type fault at that qubit.

    Args:
        H_X: X-type (or single-channel) check matrix, shape (m_X, n).
        H_Z: Z-type check matrix, shape (m_Z, n); must act on the same
            `n` qubits as `H_X`.
        H_Y: Y-type check matrix, shape (m_Y, n); must act on the same
            `n` qubits as `H_X`/`H_Z`.

    Returns:
        uint8 array: `H_X` (single mode) or the block-combined matrix of
        shape `(m_X + m_Z [+ m_Y], 2n)` (css/xyz modes).
    """
    H_X = np.asarray(H_X, dtype=np.uint8)
    m_X, n = H_X.shape
    if H_Z is None and H_Y is None:
        return H_X

    H_Z = np.asarray(H_Z, dtype=np.uint8)
    m_Z, n_Z = H_Z.shape
    assert n == n_Z, "H_X and H_Z must act on the same number of qubits."

    row_X = np.hstack([H_X, np.zeros((m_X, n), dtype=np.uint8)])
    row_Z = np.hstack([np.zeros((m_Z, n), dtype=np.uint8), H_Z])

    if H_Y is None:
        return np.vstack([row_X, row_Z])

    H_Y = np.asarray(H_Y, dtype=np.uint8)
    m_Y, n_Y = H_Y.shape
    assert n == n_Y, "H_X, H_Z, and H_Y must act on the same number of qubits."
    row_Y = np.hstack([H_Y, H_Y])
    return np.vstack([row_X, row_Z, row_Y])


def generate_space_time_pcm(
    r: int,
    H_X: np.ndarray,
    H_Z: np.ndarray | None = None,
    H_Y: np.ndarray | None = None,
    sparse: bool = True,
    perfect_first_round: bool = False,
    perfect_last_round: bool = False,
):
    """Generate the full space-time PCM over `r` rounds.

    Two block-diagonal-ish pieces, concatenated column-wise (`hstack`):
    - `data_block = I_r kron H_spatial`: one independent copy of the
      spatial check matrix per round, applied to that round's data-qubit
      fault columns (detects faults within a single round).
    - `meas_block = delta_r kron I_{m_total}`: applies the temporal
      difference (`build_delta_r`) across rounds' measurement-flip
      columns, so a measurement flip in round `i` shows up in both round
      `i` and round `i+1`'s detectors (it's undone by the *next* correct
      measurement) — the standard "detector = round diff" encoding.

    Column order of the result matches `error_model.sample_shot` /
    `build_priors`: all data columns for round 0..r-1, then all
    measurement columns for round 0..r-1.

    Args:
        r: Number of rounds.
        H_X: X-type (or single-channel) check matrix.
        H_Z: Z-type check matrix; enables "css" mode (see
            `build_spatial_matrix`).
        H_Y: Y-type check matrix; enables "xyz" mode.
        sparse: If True, return a `scipy.sparse.csr_matrix`; if False,
            return a dense uint8 `numpy.ndarray`.
        perfect_first_round: If True, model round 0's measurements as
            noiseless: its `m_total` measurement-flip columns are omitted
            entirely (the first block column of `delta_r`).
        perfect_last_round: If True, model round `r-1`'s measurements as
            noiseless, omitting the *last* block column of `delta_r`. This
            is what closes the time boundary: a measurement flip in round
            `i` normally lights up detectors `i` and `i+1`, but in the
            final round there is no detector `i+1` to catch it, so it is
            indistinguishable from a data fault. A larger code has more
            checks to flip in that unprotected round, which can make more
            rounds and bigger codes decode *worse*. A real memory
            experiment closes the boundary with a final noiseless readout
            of the data qubits; this flag models that.

    Either flag leaves `r*n_cols + meas_rounds(...)*m_total` columns. Data
    columns keep their positions in both cases, since the omitted blocks
    are inside the tail. Callers must pass the same flags to
    `error_model.sample_shot` and `error_model.build_priors` so the column
    orders still agree.

    Returns:
        The space-time PCM, mod-2 reduced, in sparse or dense form.

    Raises:
        ValueError: If idealizing leaves no noisy round (see `meas_rounds`).
    """
    # Validates the flag combination before anything is allocated.
    meas_rounds(r, perfect_first_round, perfect_last_round)

    H_spatial = build_spatial_matrix(H_X, H_Z, H_Y)
    m_total, _ = H_spatial.shape
    delta_r = build_delta_r(r)
    if perfect_first_round:
        # Round 0's measurement flips are forced to zero, so their columns
        # carry no information -- drop delta_r's first block column. The
        # round-0 detectors then depend only on round-0 data faults.
        delta_r = delta_r[:, 1:]
    if perfect_last_round:
        # Likewise for the final round, at the other end: its column is the
        # only one with no downstream detector to close it off.
        delta_r = delta_r[:, :-1]

    Hs = csr_matrix(H_spatial)
    Ir = eye(r, format="csr", dtype=np.uint8)
    Im = eye(m_total, format="csr", dtype=np.uint8)
    Dr = csr_matrix(delta_r)

    data_block = kron(Ir, Hs, format="csr")
    data_block.data %= 2
    meas_block = kron(Dr, Im, format="csr")
    meas_block.data %= 2

    H_st = hstack([data_block, meas_block], format="csr")
    H_st.data %= 2
    return H_st if sparse else np.asarray(H_st.todense(), dtype=np.uint8)


def save_pcm(H_st, path: str) -> None:
    """Save a (sparse or dense) parity-check matrix to a `.npz` file.

    Dense input is first converted to a `csr_matrix` of uint8, since
    space-time PCMs are typically large and sparse.

    Args:
        H_st: Parity-check matrix, sparse or dense array-like.
        path: Output file path, passed to `scipy.sparse.save_npz`.
    """
    H = H_st if issparse(H_st) else csr_matrix(np.asarray(H_st, dtype=np.uint8))
    save_npz(path, H.tocsr())


def load_pcm(path: str) -> csr_matrix:
    """Load a parity-check matrix previously written by `save_pcm`.

    Args:
        path: Path to a `.npz` file (as produced by `save_pcm`).

    Returns:
        The loaded matrix as a `scipy.sparse.csr_matrix`.
    """
    return load_npz(path)
