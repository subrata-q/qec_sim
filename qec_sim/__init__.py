"""Quantum error correction (QEC) space-time simulation package.

This package provides the building blocks for Monte Carlo simulation of
multi-round quantum error correction under a circuit-level noise model,
decoded with `relay_bp`:

- `spacetime_pcm`: builds the space-time parity-check matrix (PCM) that
  unrolls a single-round check matrix over `r` rounds, linking measurement
  rounds via a temporal difference matrix.
- `error_model`: defines the physical noise channel (`NoiseModel`) and
  samples fault vectors / per-column error priors for a shot.
- `logical_ops`: computes logical operator bases (over GF(2)) and checks
  whether a residual fault pattern triggers a logical failure.
- `gf2`: low-level GF(2) linear algebra (row reduction, rank, nullspace,
  quotient bases) used by `logical_ops`.
- `relay_bp_integrate`: helpers to construct and wrap `relay_bp` decoders
  for use as the `decode_fn` passed to `run_experiment`.
- `simulate`: runs the Monte Carlo experiment loop and reports the logical
  error rate with a Wilson confidence interval.
- `logging_utils`: optional per-shot logging of syndromes/corrections.

Supported code "modes", determined by which check matrices are supplied:
- `single`: one check matrix `H_X` only (e.g. classical/repetition code,
  or a single Pauli-error channel).
- `css`: `H_X` and `H_Z` (CSS code, independent X/Z error tracking).
- `xyz`: `H_X`, `H_Z`, and `H_Y` (correlated Pauli noise, e.g. XYZ codes).
"""

from .spacetime_pcm import (
    generate_space_time_pcm,
    build_spatial_matrix,
    build_delta_r,
    save_pcm,
    load_pcm,
)
from .error_model import NoiseModel, sample_shot, build_priors
from .logical_ops import compute_logical_operators, logical_failure
from .simulate import run_experiment, wilson_interval, ExperimentResult
from .logging_utils import ShotLogger, bitstring
from .relay_bp_integrate import (
    build_decoder,
    make_decode_fn,
    DEFAULT_RELAY_KWARGS,
    RELAY_KWARGS_INT,
    describe_decoder_class,
)

__all__ = [
    "generate_space_time_pcm",
    "build_spatial_matrix",
    "build_delta_r",
    "save_pcm",
    "load_pcm",
    "NoiseModel",
    "sample_shot",
    "build_priors",
    "compute_logical_operators",
    "logical_failure",
    "run_experiment",
    "wilson_interval",
    "ExperimentResult",
    "ShotLogger",
    "bitstring",
    "build_decoder",
    "make_decode_fn",
    "DEFAULT_RELAY_KWARGS",
    "RELAY_KWARGS_INT",
    "describe_decoder_class",
]
