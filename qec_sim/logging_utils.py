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
