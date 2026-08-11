# qec_sim

Simulation of multi-round quantum error correction under a
circuit-level noise model, decoded with i.e. [`relay_bp`](https://github.com/subrata-q/relay-BP). 
Use relay_bp from here https://github.com/subrata-q/relay-BP

## Install

```
pip install "qec_sim @ git+https://github.com/subrata-q/qec_sim.git"
```

## Usage

See `example_usage.py`.

## Space-time PCM layout

`generate_space_time_pcm(r, H_X, ...)` unrolls the single-round check matrix
over `r` rounds. Columns are round-major: every data-fault column for rounds
`0..r-1`, then every measurement-flip column for rounds `0..r-1`.
`sample_shot` and `build_priors` produce vectors in that same order, so all
three must be called with matching arguments.

### Perfect first round

Pass `perfect_first_round=True` to model round 0's measurements as noiseless.
Its `m_total` measurement-flip columns are dropped — they are the first block
column of the temporal difference matrix — leaving
`r*n_cols + (r-1)*m_total` columns. The round-0 detectors then depend only on
round-0 data faults.

The flag must be passed consistently to every function that indexes those
columns, since you build the PCM and priors yourself before calling
`run_experiment`:

```python
H_st   = generate_space_time_pcm(r, H_X, perfect_first_round=True)
priors = build_priors("single", n=n, r=r, m_total=m_total, noise=noise,
                      perfect_first_round=True)
result = run_experiment(H_X, r=r, decode_fn=decode_fn, perfect_first_round=True)
```

A mismatch is **not validated**. `relay_bp` accepts a priors array of any
length at construction and then panics on the first `decode` with
`pyo3_runtime.PanicException: ndarray: index out of bounds`, in both the
too-long and too-short direction. It is at least loud rather than silently
wrong, but the traceback points into ndarray and says nothing about priors, so
check the flag first when you see that panic.

Data columns keep their positions either way — the dropped block sits at the
tail — so nothing downstream of the data prefix changes.

## Per-leg solutions

Build the decoder with `collect_solutions=True` and each shot's per-leg
solutions land on the callable returned by `make_decode_fn`:

```python
decode_fn = make_decode_fn(decoder)
...
solutions, legs, converged = decode_fn.last_solutions   # None if nothing recorded
```

Add `collect_all_legs=True` to the decoder to keep the legs that *failed* to
converge as well; they appear as rows with `converged[i] == False`.

`last_solutions` only holds the most recent shot. There are two ways to keep a
whole multi-shot run:

* `make_decode_fn(decoder, keep_history=True)` appends every shot's record to
  `decode_fn.solutions_history`, so `solutions_history[i]` is shot `i`'s triple
  and `None` marks a shot that recorded nothing. Memory grows with the shot
  count — one `(rows, n)` array per shot, `rows` up to `stop_nconv`, or up to
  `num_sets + 1` under `collect_all_legs`.
* `run_experiment(solutions_dir=...)` writes one `sol_<shot>.txt` per shot, each
  with a `# leg <i> converged=<0|1>` block per recorded leg. Does not grow with
  the shot count, so prefer it for long runs.

In the default collection mode a shot that never converged writes **no**
`sol_<shot>.txt` at all. Under `collect_all_legs=True` every shot writes a
file, so that shortcut no longer holds.

`run_experiment(log_dir=...)` is the third channel: `syndromes.txt`,
`corrections.txt` and `logical_errors.txt`, one line per shot, whole-shot only.

## Scoring

`run_experiment` treats decoder non-convergence as a logical failure
unconditionally:

```python
failed = (not converged) or _check_failure(...)
```

`or` short-circuits, so the residual is never examined for a non-converged
shot. Such shots are counted in both `result.fails` and
`result.decoder_nonconvergence` (the latter a subset of the former). This is
the conservative convention, but it is pessimistic when the iteration budget is
starved — measure before trusting `p_L` from a deliberately under-resourced
run.
