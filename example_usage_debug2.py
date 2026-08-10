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

# Master seed for every random source in this script, so a run is reproducible:
#   1. the explicit_gammas draw below        (numpy Generator)
#   2. the relay decoder's gamma sampling    (seed= on the decoder; only used
#      when explicit_gammas is None, since explicit gammas replace the draw)
#   3. the noise/fault sampling per shot     (seed= on run_experiment)
# Change SEED to explore a different draw; keep it fixed to compare output files
# across runs. Min-sum BP itself has no randomness, so it needs no seed.
SEED = 9

# Code parameters (3-qubit repetition code).
H_X = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
r = 3
noise = NoiseModel(px=0.1, p_meas=0.005)

H_st = generate_space_time_pcm(r, H_X)

# print(f"H_st shape: {H_st.todense().shape}")
np.savetxt("H_st.txt", H_st.todense(), fmt="%d")

m_total = build_spatial_matrix(H_X).shape[0]
# print(f"m_total: {m_total}")
priors = build_priors("single", n=3, r=r, m_total=m_total, noise=noise)

# explicit_gammas_python = np.random.rand(
#     -1, 1, size=1, dtype=np.float32
# )  # Example explicit gammas for testing

rng = np.random.default_rng(SEED)
# explicit_gammas_python = np.array(np.random.rand(16).astype(np.float32))
explicit_gammas_python = rng.random((1, 15), dtype=np.float64)
# explicit_gammas_python = rng.uniform(low=-2, high=12, size=(1, 15))
# explicit_gammas_python = rng.integers(low=-16, high=16, size=(1, 15), endpoint=True)

print(f"Explicit gammas: {explicit_gammas_python}")

decoder_i8 = build_decoder(
    H_st,
    priors,
    decoder_class=relay_bp.RelayDecoderI8,
    max_data_value=127,  # min_data_value=-128,  # min_data_value=-128, max_data_value=127
    data_scale_value=16,  # intentionally set to 16 for checking overflow behavior
    gamma_scale_value=16,  # default value is data_scale_value
    rounding_mode="round",  # "round" (default) | "floor" | "ceiling"
    marginal_zero_error=False,  # True (default): posterior == 0 counts as an error
    collect_solutions=True,  # Save the converged solutions (up to stop_nconv)
    seed=SEED,  # Relay's gamma-sampling RNG (unused while explicit_gammas is set)
    verbose=True,
    pre_iter=30,  # number of BP iterations to run before starting the relay iterations
    set_max_iter=20,  # number of BP iterations to run per relay iteration
    stop_nconv=3,  # How many Relay ensemble solutions to find before terminating.
    num_sets=10,  # number of Relay legs
    explicit_gammas=explicit_gammas_python,
)
decode_fn = make_decode_fn(decoder_i8)

result = run_experiment(
    H_X,
    r=r,
    shots=1,
    noise=noise,
    seed=SEED,
    decode_fn=decode_fn,
    log_dir="logs/debug",
    solutions_dir="logs/debug/solutions",
    # verbose=True,
)

print(f"Shots: {result.shots}")
print(f"Failed shots: {result.fails} / {result.shots}")
# Non-convergence is always scored as a failure, so it is counted in p_L above.
# A shot counted here produces no sol_<shot>.txt, and its entry in
# corrections.txt will not satisfy the syndrome.
print(f"  of which non-converged: {result.decoder_nonconvergence}")
print(f"Logical error rate (p_L): {result.p_L:.4e}")
print(f"95% CI: ({result.ci_lo:.2e}, {result.ci_hi:.2e})")
print("Per-shot logs written to logs/debug/{syndromes,corrections,logical_errors}.txt")
print("Converged solutions written to logs/debug/solutions/sol_<shot>.txt")
print("  one block per converged relay leg, blank line between blocks,")
print("  at most stop_nconv blocks; no file at all if the shot never converged")
