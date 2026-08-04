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

# Code parameters (3-qubit repetition code).
H_X = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
r = 3
noise = NoiseModel(px=0.1, p_meas=0.005)

H_st = generate_space_time_pcm(r, H_X)

np.savetxt("H_st.txt", H_st.todense(), fmt="%d")

m_total = build_spatial_matrix(H_X).shape[0]
priors = build_priors("single", n=3, r=r, m_total=m_total, noise=noise)

decoder_i8 = build_decoder(
    H_st,
    priors,
    decoder_class=relay_bp.RelayDecoderI8,
    max_data_value=127,
    data_scale_value=16,  # intentionally set to 16 for checking overflow behavior
    # gamma_scale_value=32, # NOT ADDED
    verbose=True,
    pre_iter=1,
    set_max_iter=1,  # Keyword override
    stop_nconv=1,  # Keyword override
    num_sets=1,  # Keyword override
)
decode_fn = make_decode_fn(decoder_i8)

result = run_experiment(
    H_X,
    r=r,
    shots=1,
    noise=noise,
    seed=9,
    decode_fn=decode_fn,
    log_dir="logs/debug",
    # verbose=True,
)

print(f"Shots: {result.shots}")
print(f"Logical error rate (p_L): {result.p_L:.4e}")
print(f"95% CI: ({result.ci_lo:.2e}, {result.ci_hi:.2e})")
print("Per-shot logs written to logs/int4/{syndromes,corrections,logical_errors}.txt")
