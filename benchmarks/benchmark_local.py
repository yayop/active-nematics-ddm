"""Quick local scaling benchmark: ``python benchmarks/benchmark_local.py``."""

from __future__ import annotations

import time

from active_nematics_ddm.ddm import compute_radial_ddm
from active_nematics_ddm.lags import logarithmic_lags
from active_nematics_ddm.synthetic import translating_texture


def main() -> None:
    frames = translating_texture(n_frames=256, shape=(256, 256), seed=11)
    lags = logarithmic_lags(len(frames), maximum=64, per_decade=8)
    print("workers,elapsed_s")
    for workers in (1, 2, 4, 8, 12):
        start = time.perf_counter()
        compute_radial_ddm(
            frames,
            pixel_size_um=0.5,
            frame_interval_s=0.1,
            lags=lags,
            q_bins=48,
            max_pairs_per_lag=128,
            workers=workers,
        )
        print(f"{workers},{time.perf_counter() - start:.3f}")


if __name__ == "__main__":
    main()
