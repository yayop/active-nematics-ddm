"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from threadpoolctl import threadpool_limits

from .config import RunConfig, load_config
from .io import format_inspection
from .pipeline import is_same_or_ancestor, run_pipeline, validate_output_location
from .synthetic import write_demo


def _batch_worker(config_path: str) -> str:
    # One process per video/config; suppress nested numerical-library pools.
    with threadpool_limits(limits=1):
        config = load_config(config_path)
        output = run_pipeline(config, workers=1)
    return str(output)


def _validate_batch_configs(configs: list[RunConfig]) -> None:
    for config in configs:
        validate_output_location(config)
    for index, left in enumerate(configs):
        left_output = left.output.directory
        for right in configs[index + 1 :]:
            right_output = right.output.directory
            if is_same_or_ancestor(left_output, right_output) or is_same_or_ancestor(
                right_output, left_output
            ):
                raise ValueError(
                    f"batch output directories overlap: {left_output} and {right_output}"
                )
            for output, protected in (
                (left_output, right.input.path),
                (right_output, left.input.path),
                (left_output, right.source),
                (right_output, left.source),
            ):
                if is_same_or_ancestor(output, protected) or (
                    protected.is_dir() and is_same_or_ancestor(protected, output)
                ):
                    raise ValueError(
                        f"batch output {output} overlaps another run's input/config {protected}"
                    )
            for output, flatfield in (
                (left_output, right.input.flatfield_path),
                (right_output, left.input.flatfield_path),
            ):
                if flatfield is not None and is_same_or_ancestor(output, flatfield):
                    raise ValueError(
                        f"batch output {output} contains another run's flat-field {flatfield}"
                    )


def _demo_config(directory: Path, video: Path) -> Path:
    config = directory / "demo.toml"
    output = directory / "results"
    text = f'''# Synthetic smoke-test configuration
[input]
path = "{video.as_posix()}"
pixel_size_um = 0.5
frame_interval_s = 0.2

[analysis]
lags_per_decade = 10
max_pairs_per_lag = 128
q_bins = 40
window = "blackman-harris"
workers = 4
pair_batch_size = 8

[fit]
enabled = true
beta_min = 0.25
beta_max = 3.0
min_points = 8

[output]
directory = "{output.as_posix()}"
overwrite = false
'''
    config.write_text(text, encoding="utf-8")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="an-ddm",
        description="Differential dynamic microscopy for active-nematic videos",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect a microscopy input")
    inspect_parser.add_argument("path", type=Path)

    run_parser = subparsers.add_parser("run", help="run one TOML configuration")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--workers", type=int, help="override lag workers")

    batch_parser = subparsers.add_parser("batch", help="run independent configs in parallel")
    batch_parser.add_argument("configs", nargs="+", type=Path)
    batch_parser.add_argument("--jobs", type=int, default=4, help="concurrent videos/configs")

    demo_parser = subparsers.add_parser("demo", help="create a synthetic TIFF and config")
    demo_parser.add_argument("directory", type=Path)
    demo_parser.add_argument("--run", action="store_true", help="also execute the analysis")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            print(format_inspection(args.path))
            return 0
        if args.command == "run":
            output = run_pipeline(load_config(args.config), workers=args.workers)
            print(output)
            return 0
        if args.command == "batch":
            if args.jobs < 1:
                raise ValueError("--jobs must be positive")
            loaded = [load_config(path) for path in args.configs]
            _validate_batch_configs(loaded)
            failures = 0
            with ProcessPoolExecutor(max_workers=min(args.jobs, len(args.configs))) as pool:
                futures = {pool.submit(_batch_worker, str(path)): path for path in args.configs}
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        print(f"OK {source}: {future.result()}")
                    except Exception as error:  # keep independent runs alive
                        failures += 1
                        print(f"FAILED {source}: {error}", file=sys.stderr)
            return 1 if failures else 0
        if args.command == "demo":
            directory = args.directory.expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            video = write_demo(directory / "synthetic_translation.tif")
            config_path = _demo_config(directory, video)
            print(f"video: {video}")
            print(f"config: {config_path}")
            if args.run:
                print(f"results: {run_pipeline(load_config(config_path))}")
            return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
