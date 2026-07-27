"""Example: relay_bp decoder integration with qec_sim, and how to override
the default decoder parameters at three different scopes.

Simulates a 3-qubit repetition code (single-channel "single" mode) over
5 rounds under independent bit-flip and measurement noise, and shows:
  1. Globally overriding the package-wide relay_bp defaults.
  2. Building a floating-point decoder with a couple of per-call kwarg
     overrides on top of those defaults.
  3. Building a fixed-point decoder with its own custom default dict
     plus a per-call override, independent of the float defaults.
"""

import numpy as np
import relay_bp

from qec_sim import (
    NoiseModel,
    run_experiment,
    generate_space_time_pcm,
    build_spatial_matrix,
    build_priors,
    build_decoder,
    make_decode_fn,
    DEFAULT_RELAY_KWARGS,
    RELAY_KWARGS_INT,
)

# Option 1: Globally modify standard float defaults AND integer/fixed-point
# defaults. Since build_decoder falls back to these module-level dicts
# whenever default_kwargs/default_int_kwargs aren't passed explicitly,
# mutating them here affects every build_decoder call below (and anywhere
# else in the process) that doesn't override the same keys.
DEFAULT_RELAY_KWARGS["pre_iter"] = 60
DEFAULT_RELAY_KWARGS["num_sets"] = 80

RELAY_KWARGS_INT["max_data_value"] = 16.0
RELAY_KWARGS_INT["data_scale_value"] = 8.0

# Setup code parameters (3-qubit repetition code): H_X checks adjacent
# qubit pairs for a bit flip; single mode tracks one combined fault
# channel (via NoiseModel.px) rather than independent X/Y/Z.
H_X = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
r = 5
noise = NoiseModel(px=0.005, p_meas=0.005)

# Space-time PCM (detectors from data + measurement faults over r rounds)
# and matching per-column error priors for the decoder.
H_st = generate_space_time_pcm(r, H_X)
m_total = build_spatial_matrix(H_X).shape[0]
priors = build_priors("single", n=3, r=r, m_total=m_total, noise=noise)

# Option 2: Floating-point decoder (uses modified DEFAULT_RELAY_KWARGS + kwarg overrides)
decoder_f32 = build_decoder(
    H_st,
    priors,
    decoder_class=relay_bp.RelayDecoderF32,
    gamma0=0.70,  # Keyword override
    set_max_iter=40,  # Keyword override
)

# Option 3: Fixed-point decoder (uses RELAY_KWARGS_INT or explicit custom_int_kwargs)
custom_int_defaults = {
    "max_data_value": 32.0,
    "data_scale_value": 16.0,
}

decoder_i8 = build_decoder(
    H_st,
    priors,
    decoder_class=getattr(relay_bp, "RelayDecoderI8", relay_bp.RelayDecoderF32),
    default_int_kwargs=custom_int_defaults,  # Pass custom integer dictionary
    max_data_value=64.0,  # Keyword override for fixed-point param
)

# Run the Monte Carlo experiment using the floating-point decoder built above.
# (decoder_i8 is built to demonstrate Option 3 but isn't run here.)
decode_fn = make_decode_fn(decoder_f32)
result = run_experiment(H_X, r=r, shots=2000, noise=noise, seed=4, decode_fn=decode_fn)

print(f"Shots: {result.shots}")
print(f"Logical error rate (p_L): {result.p_L:.4e}")
print(f"95% CI: ({result.ci_lo:.2e}, {result.ci_hi:.2e})")
