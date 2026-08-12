"""Same as example_usage_debug.py, but prints the leg solutions instead of
saving them. No solutions_dir is passed; they are read off the decode_fn.
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
)

# import time

SEED = 34  # time.time_ns() % 2**32  # or any fixed integer, e.g. 9

# Code parameters (3-qubit repetition code).
H_X = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
r = 3
noise = NoiseModel(px=0.01, p_meas=0.005)

H_st = generate_space_time_pcm(r, H_X)

m_total = build_spatial_matrix(H_X).shape[0]
priors = build_priors("single", n=3, r=r, m_total=m_total, noise=noise)

rng = np.random.default_rng(SEED)
explicit_gammas_python = rng.random((1, 15), dtype=np.float64)

print(f"Explicit gammas: {explicit_gammas_python}")

decoder_i8 = build_decoder(
    H_st,
    priors,
    decoder_class=relay_bp.RelayDecoderI8,
    max_data_value=127,
    data_scale_value=16,
    gamma_scale_value=16,
    rounding_mode="round",
    marginal_zero_error=False,
    collect_solutions=True,  # required for the solutions to be kept
    collect_all_legs=True,  # keep the non-converged legs too, with their flag
    seed=SEED,
    verbose=True,  # the per-iteration trace would drown the solutions below
    pre_iter=15,
    set_max_iter=10,
    stop_nconv=3,
    num_sets=10,
)
decode_fn = make_decode_fn(decoder_i8, keep_history=True)

result = run_experiment(
    H_X,
    r=r,
    shots=2,
    noise=noise,
    seed=SEED,
    decode_fn=decode_fn,
    log_dir="logs/debug2",
    solutions_dir="logs/debug2/solutions",
)

# keep_history=True keep all per-leg solutions for all shots
print("Per-leg solutions:")

for shot, rec in enumerate(decode_fn.solutions_history):
    if rec is None:
        print(f"shot {shot}: nothing recorded")
        continue
    solutions, legs, converged = rec
    print(f"shot {shot}")
    for leg, ok, row in zip(
        np.asarray(legs), np.asarray(converged), np.asarray(solutions)
    ):
        flag = "converged" if ok else "  failed "
        print(f"        leg {leg} {flag}: {''.join(str(int(b)) for b in row)}")


print()
print(f"Shots: {result.shots}")
print(f"Failed shots: {result.fails} / {result.shots}")
print(f"  of which non-converged: {result.decoder_nonconvergence}")
print(f"Logical error rate (p_L): {result.p_L:.4e}")
