from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import ModelConfig
from .pipeline import generate_model


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="Generate a triply periodic Voronoi sheet aerogel model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration file (default: built-in settings).",
    )
    parser.add_argument("--output", type=Path, help="Override output directory.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use lighter sampling/voxel settings for a fast diagnostic run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """读取配置、应用少量命令行覆盖项并启动完整生成流程。"""

    args = _parser().parse_args(argv)
    config = ModelConfig.from_json(args.config) if args.config else ModelConfig()
    if args.output is not None:
        config = replace(config, output_dir=str(args.output))
    if args.seed is not None:
        config = replace(config, random_seed=args.seed)
    if args.quick:
        # 快速模式只用于检查拓扑和工作流，不应替代最终网格收敛计算。
        config = replace(
            config,
            qmc_power=min(config.qmc_power, 16),
            validation_qmc_power=min(config.validation_qmc_power, 15),
            voxel_size_nm=max(config.voxel_size_nm, config.thickness_nm),
        )
    config.validate()
    result = generate_model(config, progress=print)
    print(f"Model written to: {result.output_dir}")
    print(f"QMC porosity: {100.0 * result.qmc_porosity:.4f}%")
    print(f"Voxel porosity: {100.0 * result.voxel_porosity:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
