"""Integration helpers for using `relay_bp` decoders with `qec_sim`.

Bridges qec_sim's space-time PCM / priors representation to `relay_bp`'s
decoder constructors, and wraps a decoder instance into the plain
`decode_fn(detectors) -> (correction, converged)` callable expected by
`simulate.run_experiment`.
"""

from __future__ import annotations
import inspect
import numpy as np
from scipy.sparse import csr_matrix
import relay_bp

# Default relay_bp keyword arguments for floating-point decoder classes
# (e.g. RelayDecoderF32). Tuned as reasonable general-purpose defaults;
# override via **relay_kwargs in build_decoder or by mutating this dict.
DEFAULT_RELAY_KWARGS: dict = {
    "gamma0": 0.65,
    "pre_iter": 80,
    "num_sets": 100,
    "set_max_iter": 60,
    "gamma_dist_interval": (-0.24, 0.66),
    "stop_nconv": 5,
}

# Additional keyword arguments merged on top of DEFAULT_RELAY_KWARGS only
# for fixed-point decoder classes (RelayDecoderI8 / RelayDecoderI32),
# which quantize messages and need explicit scale/clamp parameters.
RELAY_KWARGS_INT: dict = {
    "max_data_value": 8.0,
    "data_scale_value": 4.0,
}


def describe_decoder_class(decoder_class) -> str:
    """Best-effort introspection of a relay_bp decoder class's constructor.

    Useful for interactively discovering what keyword arguments a given
    `decoder_class` accepts, since relay_bp decoders are typically
    compiled/pybind11 classes whose signatures aren't always inspectable.

    Args:
        decoder_class: A relay_bp decoder class (not instance).

    Returns:
        A string starting with "signature: " if `inspect.signature`
        succeeds, else "docstring:\\n<doc>" if a docstring is found on the
        class or its `__new__`, else "signature unavailable".
    """
    try:
        return f"signature: {inspect.signature(decoder_class)}"
    except (ValueError, TypeError):
        pass
    doc = decoder_class.__doc__ or getattr(
        getattr(decoder_class, "__new__", None), "__doc__", None
    )
    return f"docstring:\n{doc}" if doc else "signature unavailable"


def build_decoder(
    H_spacetime,
    priors: np.ndarray,
    decoder_class=None,
    default_kwargs: dict | None = None,
    default_int_kwargs: dict | None = None,
    **relay_kwargs,
):
    """Instantiate a relay_bp decoder for a space-time PCM and priors.

    Merges keyword arguments in increasing priority: `default_kwargs`
    (defaults to `DEFAULT_RELAY_KWARGS`), then — only if `decoder_class` is
    one of the fixed-point classes (`RelayDecoderI8`/`RelayDecoderI32`) —
    `default_int_kwargs` (defaults to `RELAY_KWARGS_INT`), then any
    explicit `**relay_kwargs` overrides. This lets callers globally tweak
    defaults, swap in a custom fixed-point default set, and/or override
    individual parameters per call, all without editing this function.

    Args:
        H_spacetime: Space-time PCM, dense array or scipy sparse matrix
            (converted to `csr_matrix` of uint8 if not already sparse).
        priors: Per-column error priors (see `error_model.build_priors`).
        decoder_class: relay_bp decoder class to instantiate; defaults to
            `relay_bp.RelayDecoderF32`.
        default_kwargs: Base kwargs to use instead of `DEFAULT_RELAY_KWARGS`.
        default_int_kwargs: Fixed-point kwargs to use instead of
            `RELAY_KWARGS_INT`, applied only for fixed-point decoder
            classes.
        **relay_kwargs: Per-call overrides, highest priority.

    Returns:
        An instantiated `decoder_class` object.

    Raises:
        TypeError: If `decoder_class` rejects the merged kwargs; the
            error is re-raised with the offending kwarg names attached.
    """
    if decoder_class is None:
        decoder_class = relay_bp.RelayDecoderF32

    base_defaults = DEFAULT_RELAY_KWARGS if default_kwargs is None else default_kwargs
    int_defaults = (
        RELAY_KWARGS_INT if default_int_kwargs is None else default_int_kwargs
    )

    is_fixed_point = decoder_class in (
        getattr(relay_bp, "RelayDecoderI8", object()),
        getattr(relay_bp, "RelayDecoderI32", object()),
    )

    kwargs = {
        **base_defaults,
        **(int_defaults if is_fixed_point else {}),
        **relay_kwargs,
    }

    H = (
        csr_matrix(np.asarray(H_spacetime, dtype=np.uint8))
        if not hasattr(H_spacetime, "tocsr")
        else H_spacetime.astype(np.uint8)
    )
    try:
        return decoder_class(
            H, error_priors=np.asarray(priors, dtype=np.float64), **kwargs
        )
    except TypeError as e:
        raise TypeError(
            f"{decoder_class.__name__} failed with kwargs: {sorted(kwargs)}: {e}"
        ) from e


def make_decode_fn(decoder):
    """Wrap a relay_bp decoder instance for use as `run_experiment`'s `decode_fn`.

    Args:
        decoder: An instantiated relay_bp decoder (e.g. from `build_decoder`)
            exposing a `decode_detailed(detectors)` method.

    Returns:
        Callable `decode_fn(detectors) -> (correction, converged)` where
        `correction` is a uint8 array over all space-time PCM columns and
        `converged` is whether the decoder reports success.
    """

    def decode_fn(detectors):
        result = decoder.decode_detailed(detectors.astype(np.uint8))
        return np.asarray(result.decoding, dtype=np.uint8), bool(result.success)

    return decode_fn
