from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import NanowireConfig
from .pipeline import generate_nanowire_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a periodic 3-D Voronoi nanowire display model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration file (default: built-in settings).",
    )
    parser.add_argument("--output", type=Path, help="Override output directory.")
    parser.add_argument("--seed", type=int, help="Override the random seed.")
    parser.add_argument(
        "--porosity",
        type=float,
        help="Override target porosity as a fraction, for example 0.97.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use lighter sampling and display settings for a diagnostic run.",
    )
    parser.add_argument(
        "--stl",
        action="store_true",
        help="Also export a visual, non-Boolean-unioned nanowire STL (nm coordinates).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = NanowireConfig.from_json(args.config) if args.config else NanowireConfig()
    if args.output is not None:
        config = replace(config, output_dir=str(args.output))
    if args.seed is not None:
        config = replace(config, random_seed=args.seed)
    if args.porosity is not None:
        config = replace(config, target_porosity=args.porosity)
    if args.quick:
        config = replace(
            config,
            qmc_power=min(config.qmc_power, 15),
            validation_qmc_power=min(config.validation_qmc_power, 16),
            tube_sides=min(config.tube_sides, 8),
        )
    config.validate()
    result = generate_nanowire_model(config, export_stl=args.stl, progress=print)
    print(f"Model written to: {result.output_dir}")
    if result.stl_path is not None:
        print(f"Display STL written to: {result.stl_path}")
    print(f"Control porosity: {100.0 * result.control_porosity:.4f}%")
    print(f"Validation porosity: {100.0 * result.validation_porosity:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
