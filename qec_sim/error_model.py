"""Physical noise model and per-shot Monte Carlo fault sampling.

Independent depolarizing-style Pauli errors (rates `px`, `py`, `pz`) are
sampled per data qubit per round, plus independent measurement flips
(rate `p_meas`) per check per round. A Y error is treated as a correlated
X-and-Z fault on the same qubit/round, which is why `ex`/`ez` are derived
from `is_x`/`is_y`/`is_z` rather than sampled independently.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class NoiseModel:
    """Physical error rates for the Pauli and measurement channels.

    Attributes:
        px: Per-qubit, per-round probability of an X error.
        py: Per-qubit, per-round probability of a Y error.
        pz: Per-qubit, per-round probability of a Z error.
        p_meas: Per-check, per-round probability of a measurement flip.

    In "single" mode (see `sample_shot`/`build_priors`), only `px` is used
    and represents the combined fault rate for the single tracked channel;
    `py` and `pz` must be left at 0 in that case.
    """

    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0
    p_meas: float = 0.0

    def __post_init__(self):
        if self.px + self.py + self.pz > 1.0:
            raise ValueError("px + py + pz must not exceed 1.0.")


def _single_mode_prob(noise: NoiseModel) -> float:
    """Return the combined fault rate for single-channel mode.

    Raises:
        ValueError: If `py` or `pz` is nonzero, since single mode only
            tracks one error channel and expects the combined rate in `px`.
    """
    if noise.py or noise.pz:
        raise ValueError("Single mode requires py=pz=0. Pass combined fault rate as px.")
    return noise.px


def _sample_ez_ex(rng: np.random.Generator, n: int, r: int, px: float, py: float, pz: float):
    """Sample independent X/Z fault indicator arrays for CSS/XYZ modes.

    Draws one uniform random number per (round, qubit) and partitions the
    unit interval into disjoint X/Y/Z outcome bands so that at most one of
    X, Y, Z fires per site. A Y outcome sets both the X and Z indicators
    (`ex`/`ez`), matching Y = X compose Z.

    Args:
        rng: NumPy random generator.
        n: Number of data qubits.
        r: Number of rounds.
        px, py, pz: Per-site error probabilities for each Pauli channel.

    Returns:
        Tuple `(ez, ex)` of `(r, n)` uint8 arrays, one entry per
        (round, qubit) indicating whether a Z-type / X-type fault fired.
    """
    u = rng.random((r, n))
    is_x = u < px
    is_y = (u >= px) & (u < px + py)
    is_z = (u >= px + py) & (u < px + py + pz)
    ex = (is_x | is_y).astype(np.uint8)
    ez = (is_z | is_y).astype(np.uint8)
    return ez, ex


def sample_shot(
    rng: np.random.Generator,
    mode: str,
    n: int,
    r: int,
    m_total: int,
    noise: NoiseModel,
    perfect_first_round: bool = False,
):
    """Sample one Monte Carlo shot: a full space-time fault vector.

    Draws data-qubit faults for every round (per `mode`, see module
    docstring) plus independent measurement flips per check per round,
    and flattens them in the same round-major column order used by
    `spacetime_pcm.generate_space_time_pcm` (all data columns for round 0,
    then round 1, ..., followed by all measurement columns for round 0,
    then round 1, ...).

    Args:
        rng: NumPy random generator.
        mode: One of "single", "css", or "xyz" (see package docstring).
        n: Number of data qubits.
        r: Number of rounds.
        m_total: Total number of checks per round (rows of the spatial
            check matrix), used to size the measurement-flip block.
        noise: Physical error rates to sample from.
        perfect_first_round: If True, round 0's measurements are noiseless
            and no flips are drawn for it, so the measurement block covers
            rounds 1..r-1 only. Must match the flag passed to
            `spacetime_pcm.generate_space_time_pcm`.

    Returns:
        Tuple `(e_full, true_residual)`:
            e_full: uint8 array of length `r*n_cols + r_meas*m_total`
                (data faults across all rounds, then measurement flips
                across all noisy rounds, where `r_meas` is `r-1` under
                `perfect_first_round` and `r` otherwise) — the full
                space-time fault vector.
            true_residual: dict mapping channel name(s) to the true
                round-parity residual per qubit (XOR of that channel's
                fault across all rounds), used later to score logical
                failure against the decoder's correction. Keys are
                `{"single"}` in single mode or `{"ez", "ex"}` otherwise.
    """
    if mode == "single":
        p = _single_mode_prob(noise)
        data = (rng.random((r, n)) < p).astype(np.uint8)
        data_flat = data.reshape(-1)
        true_residual = {"single": (data.sum(axis=0) % 2).astype(np.uint8)}
    else:
        ez, ex = _sample_ez_ex(rng, n, r, noise.px, noise.py, noise.pz)
        per_round = np.concatenate([ez, ex], axis=1)
        data_flat = per_round.reshape(-1)
        true_residual = {
            "ez": (ez.sum(axis=0) % 2).astype(np.uint8),
            "ex": (ex.sum(axis=0) % 2).astype(np.uint8),
        }

    r_meas = r - 1 if perfect_first_round else r
    meas = (rng.random((r_meas, m_total)) < noise.p_meas).astype(np.uint8)
    meas_flat = meas.reshape(-1)

    e_full = np.concatenate([data_flat, meas_flat]).astype(np.uint8)
    return e_full, true_residual


def build_priors(
    mode: str,
    n: int,
    r: int,
    m_total: int,
    noise: NoiseModel,
    perfect_first_round: bool = False,
) -> np.ndarray:
    """Build the per-column prior fault probabilities for the decoder.

    Produces one prior probability per column of the space-time PCM
    (see `spacetime_pcm.generate_space_time_pcm`), in the same
    data-then-measurement, round-major column order as `sample_shot`.
    In CSS/XYZ mode, the Z-type prior is `pz + py` and the X-type prior
    is `px + py`, since a Y error contributes to both channels.

    Args:
        mode: One of "single", "css", or "xyz" (see package docstring).
        n: Number of data qubits.
        r: Number of rounds.
        m_total: Total number of checks per round.
        noise: Physical error rates to derive priors from.
        perfect_first_round: If True, round 0 contributes no
            measurement-flip priors, matching the columns dropped by
            `spacetime_pcm.generate_space_time_pcm` under the same flag.

    Returns:
        1D float64 array of priors, one per space-time PCM column.
    """
    if mode == "single":
        p = _single_mode_prob(noise)
        per_round = np.full(n, p, dtype=np.float64)
    else:
        ez_prior = noise.pz + noise.py
        ex_prior = noise.px + noise.py
        per_round = np.concatenate([np.full(n, ez_prior), np.full(n, ex_prior)])
    data_priors = np.tile(per_round, r)
    r_meas = r - 1 if perfect_first_round else r
    meas_priors = np.full(r_meas * m_total, noise.p_meas, dtype=np.float64)
    return np.concatenate([data_priors, meas_priors])
