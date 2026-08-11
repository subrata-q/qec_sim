"""Per-shot text logging for syndromes, decoder corrections, and outcomes."""

from __future__ import annotations
from pathlib import Path
import numpy as np


def bitstring(arr) -> str:
    """Flatten a binary array into a contiguous "0101..." string.

    Args:
        arr: Array-like of 0/1 values (any shape; flattened in row-major
            order via `ravel`).

    Returns:
        String of '0'/'1' characters, one per element.
    """
    return "".join(str(int(b)) for b in np.asarray(arr).ravel())


class SolutionLogger:
    """Writes one shot's per-leg solutions to its own file.

    Creates `sol_<shot>.txt` under `solutions_dir`, 1-indexed, so shot 1 lands
    in `sol_1.txt`. Inside a file there is one block per recorded relay leg —
    a `# leg <i> converged=<0|1>` header followed by the solution bitstring —
    with a blank line between blocks.

    By default the decoder records only converged legs, so every block reads
    `converged=1`, there are at most `stop_nconv` of them, and a shot that
    never converged produces **no file at all** — a missing `sol_<shot>.txt`
    means that shot found no solution. Build the decoder with
    `collect_all_legs=True` to get a block for every leg that ran, including
    the failed ones (`converged=0`); every shot then writes a file.
    """

    def __init__(self, solutions_dir: str):
        """Create `solutions_dir` and clear any `sol_*.txt` left in it.

        Stale files must go: a shot that does not converge writes no file, so
        without this a previous run's `sol_<shot>.txt` would survive next to a
        fresh `corrections.txt` and look like it belonged to the current run.
        This mirrors `ShotLogger` opening its files with mode "w".

        Args:
            solutions_dir: Directory to write `sol_<shot>.txt` files into.
        """
        self.solutions_dir = Path(solutions_dir)
        self.solutions_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.solutions_dir.glob("sol_*.txt"):
            stale.unlink()

    def log(self, shot_index: int, solutions, legs, converged=None) -> None:
        """Write one shot's recorded solutions, one block per leg.

        Writes nothing when there are no solutions — either the decoder was
        not built with `collect_solutions=True`, or it recorded converged legs
        only (the default) and this shot never converged.

        Args:
            shot_index: 0-based shot number; the file is named with
                `shot_index + 1`.
            solutions: (rows, n) array of solutions, one row per recorded leg,
                as returned by `DecodeResult.solutions`.
            legs: Length-`rows` array giving each row's relay leg, as returned
                by `DecodeResult.solution_legs`.
            converged: Length-`rows` bool array flagging which rows came from
                a leg that converged, as returned by
                `DecodeResult.solution_converged`. `None` is treated as
                all-converged, which is what the default collection mode
                records.
        """
        if solutions is None or legs is None:
            return

        solutions = np.atleast_2d(np.asarray(solutions))
        legs = np.asarray(legs).ravel()
        if legs.size == 0:
            # Nothing recorded for this shot: leave no file behind.
            return
        flags = (
            np.ones(legs.size, dtype=bool)
            if converged is None
            else np.asarray(converged).ravel().astype(bool)
        )

        blocks = [
            f"# leg {leg} converged={int(ok)}\n{bitstring(row)}"
            for leg, ok, row in zip(legs.tolist(), flags.tolist(), solutions)
        ]
        path = self.solutions_dir / f"sol_{shot_index + 1}.txt"
        path.write_text("\n\n".join(blocks) + "\n")


class ShotLogger:
    """Writes one line per Monte Carlo shot to three parallel text files.

    Files are created under `log_dir`: `syndromes.txt` (detector bits),
    `corrections.txt` (decoder's full correction bits), and
    `logical_errors.txt` ("1"/"0" outcome flag) — line `i` in each file
    corresponds to the same shot. Supports use as a context manager to
    ensure files are closed.
    """

    def __init__(self, log_dir: str, mode: str = "w"):
        """Open the three log files under `log_dir`, creating it if needed.

        Args:
            log_dir: Directory to write `syndromes.txt`, `corrections.txt`,
                and `logical_errors.txt` into.
            mode: File open mode for all three files (e.g. "w" to
                overwrite, "a" to append).
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._syn_f = open(self.log_dir / "syndromes.txt", mode)
        self._corr_f = open(self.log_dir / "corrections.txt", mode)
        self._log_f = open(self.log_dir / "logical_errors.txt", mode)

    def log(self, detectors, decoded_full, logical_error: bool) -> None:
        """Append one line to each log file for a single shot.

        Args:
            detectors: Detector (syndrome) bit array for the shot.
            decoded_full: Decoder's full space-time correction bit array.
            logical_error: Whether this shot resulted in a logical failure.
        """
        self._syn_f.write(bitstring(detectors) + "\n")
        self._corr_f.write(bitstring(decoded_full) + "\n")
        self._log_f.write(("1" if logical_error else "0") + "\n")

    def close(self) -> None:
        """Close all three underlying log files."""
        self._syn_f.close()
        self._corr_f.close()
        self._log_f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
