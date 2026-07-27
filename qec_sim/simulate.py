"""Multi-round QEC Monte Carlo experiment execution.

Ties together `spacetime_pcm` (PCM construction), `error_model` (fault
sampling), `logical_ops` (failure detection), and a caller-supplied
decoder (see `relay_bp_integrate`) to run a shot-by-shot Monte Carlo loop
and report the logical error rate with a confidence interval.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .spacetime_pcm import generate_space_time_pcm, build_spatial_matrix
from .error_model import NoiseModel, sample_shot
from .logical_ops import compute_logical_operators, logical_failure
from .logging_utils import ShotLogger


def _mode_of(H_Z, H_Y) -> str:
    """Infer the code mode ("single"/"css"/"xyz") from which matrices are given."""
    if H_Z is None and H_Y is None:
        return "single"
    if H_Y is None:
        return "css"
    return "xyz"


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Compute a Wilson score confidence interval for a binomial proportion.

    Preferred over the normal (Wald) approximation for rare-event rates
    like logical error rates, since it stays well-behaved (bounded in
    [0, 1], sane width) even when `k` is small or zero.

    Args:
        k: Number of successes (e.g. failed shots).
        n: Number of trials (e.g. total shots).
        z: Z-score for the desired confidence level; default 1.96 is the
            two-sided 95% critical value.

    Returns:
        Tuple `(phat, ci_lo, ci_hi)`: the point estimate `k/n` and the
        lower/upper confidence bounds, clipped to `[0, 1]`. Returns
        `(0.0, 0.0, 1.0)` if `n == 0` (undefined proportion).
    """
    if n == 0:
        return 0.0, 0.0, 1.0
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


@dataclass
class ExperimentResult:
    """Statistical summary of a `run_experiment` Monte Carlo run.

    Attributes:
        shots: Total number of shots simulated.
        fails: Number of shots that resulted in logical failure (includes
            decoder non-convergence, which is always counted as a failure).
        p_L: Point-estimate logical error rate, `fails / shots`.
        ci_lo: Lower bound of the 95% Wilson confidence interval on `p_L`.
        ci_hi: Upper bound of the 95% Wilson confidence interval on `p_L`.
        decoder_nonconvergence: Number of shots where the decoder itself
            reported non-convergence (a subset of `fails`).
        mode: Code mode used, "single", "css", or "xyz".
        per_shot_success: Boolean array, one entry per shot, True where
            the shot succeeded (excluded from `repr` since it's large).
    """

    shots: int
    fails: int
    p_L: float
    ci_lo: float
    ci_hi: float
    decoder_nonconvergence: int
    mode: str
    per_shot_success: np.ndarray = field(repr=False)


def _decoded_residual(decoded_full: np.ndarray, mode: str, n: int, r: int):
    """Collapse the decoder's full space-time correction to a per-qubit residual.

    Sums (mod 2) the decoded data-fault columns across all `r` rounds for
    each channel, mirroring how `sample_shot` derives `true_residual` — so
    the two can be XORed together to get the net uncorrected fault.

    Args:
        decoded_full: Decoder's full correction vector over all space-time
            PCM columns (data columns for all rounds, then measurement
            columns for all rounds); only the data-column prefix is used.
        mode: Code mode, "single", "css", or "xyz".
        n: Number of data qubits.
        r: Number of rounds.

    Returns:
        Dict of per-channel round-parity residuals, same shape/keys as
        `error_model.sample_shot`'s `true_residual` output.
    """
    n_cols = n if mode == "single" else 2 * n
    data_part = np.asarray(decoded_full[: r * n_cols], dtype=np.uint8).reshape(r, n_cols)
    if mode == "single":
        return {"single": (data_part.sum(axis=0) % 2).astype(np.uint8)}
    ez = data_part[:, :n]
    ex = data_part[:, n : 2 * n]
    return {
        "ez": (ez.sum(axis=0) % 2).astype(np.uint8),
        "ex": (ex.sum(axis=0) % 2).astype(np.uint8),
    }


def _check_failure(
    mode: str, true_residual: dict, decoded_residual: dict, logical_ops: dict
) -> bool:
    """Combine true and decoded residuals and test against logical operators.

    For each channel, XORs the true fault residual with the decoder's
    correction residual to get the net uncorrected fault, then delegates
    to `logical_ops.logical_failure`. In "css" mode, X and Z channels are
    checked independently and either failing counts as failure; in "xyz"
    mode the two channels are concatenated and checked jointly against a
    combined logical basis (they aren't independent under XYZ noise).

    Args:
        mode: Code mode, "single", "css", or "xyz".
        true_residual: True fault residual per channel (see `sample_shot`).
        decoded_residual: Decoder's correction residual per channel (see
            `_decoded_residual`).
        logical_ops: Logical operator basis dict from
            `logical_ops.compute_logical_operators`.

    Returns:
        True if the net residual triggers a logical failure.
    """
    if mode == "single":
        residual = (true_residual["single"] ^ decoded_residual["single"]) % 2
        return logical_failure(residual, logical_ops["logical"])
    if mode == "css":
        r_ez = (true_residual["ez"] ^ decoded_residual["ez"]) % 2
        r_ex = (true_residual["ex"] ^ decoded_residual["ex"]) % 2
        return logical_failure(r_ez, logical_ops["logical_z"]) or logical_failure(
            r_ex, logical_ops["logical_x"]
        )

    r_ez = (true_residual["ez"] ^ decoded_residual["ez"]) % 2
    r_ex = (true_residual["ex"] ^ decoded_residual["ex"]) % 2
    residual = np.concatenate([r_ez, r_ex])
    return logical_failure(residual, logical_ops["logical"])


def run_experiment(
    H_X,
    H_Z=None,
    H_Y=None,
    r: int = 5,
    shots: int = 1000,
    noise: NoiseModel = NoiseModel(),
    seed: int | None = None,
    decode_fn=None,
    log_dir: str | None = None,
) -> ExperimentResult:
    """Run a Monte Carlo simulation of a multi-round space-time QEC code.

    For each of `shots` trials: samples a fault vector and its true
    residual (`error_model.sample_shot`), propagates it through the
    space-time PCM to get the detector (syndrome) pattern, decodes it with
    `decode_fn`, and checks whether the true fault XOR the decoder's
    correction triggers a logical failure (`logical_ops.logical_failure`).
    Decoder non-convergence is always treated as a failure regardless of
    whether the residual would otherwise have been correctable.

    Args:
        H_X: X-type (or single-channel) check matrix, shape (checks, n).
        H_Z: Z-type check matrix; supplying it (without `H_Y`) selects
            "css" mode.
        H_Y: Y-type check matrix; supplying it selects "xyz" mode.
        r: Number of QEC rounds per shot.
        shots: Number of Monte Carlo trials to run.
        noise: Physical noise model to sample faults from.
        seed: Seed for the RNG; `None` uses OS entropy (non-reproducible).
        decode_fn: Callable `detectors -> correction` or
            `detectors -> (correction, converged)`, e.g. from
            `relay_bp_integrate.make_decode_fn`. If only a correction is
            returned, convergence is assumed True. Required.
        log_dir: If given, write per-shot syndromes/corrections/outcomes
            to this directory via `logging_utils.ShotLogger`.

    Returns:
        `ExperimentResult` summarizing the run.

    Raises:
        ValueError: If `decode_fn` is not provided.
    """
    if decode_fn is None:
        raise ValueError("run_experiment requires a valid decode_fn.")

    H_X = np.asarray(H_X, dtype=np.uint8)
    mode = _mode_of(H_Z, H_Y)
    n = H_X.shape[1]
    H_spatial = build_spatial_matrix(H_X, H_Z, H_Y)
    m_total = H_spatial.shape[0]

    H_st = generate_space_time_pcm(r, H_X, H_Z, H_Y, sparse=True)
    logical_ops = compute_logical_operators(H_X, H_Z, H_Y)

    _user_decode_fn = decode_fn

    def decode_fn_wrapper(detectors):
        # Normalize decode_fn's return value to always be (correction, converged).
        out = _user_decode_fn(detectors)
        if isinstance(out, tuple):
            return np.asarray(out[0], dtype=np.uint8), bool(out[1])
        return np.asarray(out, dtype=np.uint8), True

    rng = np.random.default_rng(seed)
    per_shot_success = np.zeros(shots, dtype=bool)
    nonconvergence = 0

    logger = ShotLogger(log_dir) if log_dir is not None else None
    try:
        for s in range(shots):
            e_full, true_residual = sample_shot(rng, mode, n, r, m_total, noise)
            # Detector (syndrome) pattern triggered by this shot's faults.
            detectors = (H_st @ e_full) % 2
            decoded_full, converged = decode_fn_wrapper(detectors)
            if not converged:
                nonconvergence += 1
            decoded_residual = _decoded_residual(decoded_full, mode, n, r)
            failed = (not converged) or _check_failure(
                mode, true_residual, decoded_residual, logical_ops
            )
            per_shot_success[s] = not failed
            if logger is not None:
                logger.log(detectors, decoded_full, failed)
    finally:
        if logger is not None:
            logger.close()

    fails = shots - int(per_shot_success.sum())
    p_L, lo, hi = wilson_interval(fails, shots)
    return ExperimentResult(
        shots=shots,
        fails=fails,
        p_L=p_L,
        ci_lo=lo,
        ci_hi=hi,
        decoder_nonconvergence=nonconvergence,
        mode=mode,
        per_shot_success=per_shot_success,
    )
