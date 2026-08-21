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

SEED = 11

# 3-qubit repetition code: each check compares an adjacent pair of qubits.
# "single" mode tracks one combined fault channel (NoiseModel.px) rather than
# independent X/Y/Z.
H_X = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
n = H_X.shape[1]
r = 4
noise = NoiseModel(px=0.02, p_meas=0.02)
shots = 3000

PERFECT_LAST_ROUND = True

m_total = build_spatial_matrix(H_X).shape[0]
H_st = generate_space_time_pcm(r, H_X, perfect_last_round=PERFECT_LAST_ROUND)
np.savetxt("H_st.txt", H_st.todense(), fmt="%d")
priors = build_priors(
    "single",
    n=n,
    r=r,
    m_total=m_total,
    noise=noise,
    perfect_last_round=PERFECT_LAST_ROUND,
)

DECODER_CLASS = relay_bp.RelayDecoderI8

decoder = build_decoder(
    H_st,
    priors,
    decoder_class=DECODER_CLASS,
    max_data_value=127.0,
    symmetric_range=True,
    data_scale_value=8.0,
    gamma_scale_value=8.0,
    rounding_mode="round",
    gamma0=0.65,
    pre_iter=80,
    set_max_iter=60,
    num_sets=100,
    stop_nconv=5,
    gamma_dist_interval=(-0.24, 0.66),
    seed=SEED,
    # explicit_gammas=explicit_gammas_python,
)

result = run_experiment(
    H_X,
    r=r,
    shots=shots,
    noise=noise,
    seed=SEED,
    decode_fn=make_decode_fn(decoder),
    perfect_last_round=PERFECT_LAST_ROUND,
)

print(f"{DECODER_CLASS.__name__}, d=3 repetition code, {r} rounds")
print(f"  noise: px={noise.px}, p_meas={noise.p_meas}")
print(f"  space-time PCM: {H_st.shape[0]} detectors x {H_st.shape[1]} columns")
print(f"Shots: {result.shots}")
print(f"Failed shots: {result.fails} / {result.shots}")
# Non-convergence is always scored as a failure, so it is already inside p_L.
print(f"  of which non-converged: {result.decoder_nonconvergence}")
print(f"Logical error rate (p_L): {result.p_L:.4e}")
print(f"95% CI: ({result.ci_lo:.2e}, {result.ci_hi:.2e})")
