"""修改下面的参数，然后按 F5 运行整个文件。"""

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from voronoi_nanowire import (  # noqa: E402
    NanowireConfig,
    export_cached_display_stl,
    generate_nanowire_model,
)

STEP = "image"  # 第一步用 "image"；第二步改成 "stl" 后再运行。


def main() -> None:
    # 长度单位为 nm；孔隙率用小数，0.95 表示 95%。
    config = NanowireConfig(
        cube_size_nm=1000.0,
        num_seeds=400,
        seed_min_distance_factor=0.30,
        rod_diameter_nm=20.0,
        target_porosity=0.95,
        porosity_tolerance=0.005,
        random_seed=42,
        qmc_power=18,
        validation_qmc_power=19,
        tube_sides=12,
        output_dir=str(PROJECT_DIR / "output" / "spyder_nanowire"),
    )
    if STEP == "image":
        result = generate_nanowire_model(
            config, save_display_mesh=True, progress=print
        )
        print(f"演示图片：{result.preview_path}")
        print(f"已保存结构网格：{result.mesh_path}")
        print(f"独立验证孔隙率：{100.0 * result.validation_porosity:.4f}%")
        if (Path(config.output_dir) / "nanowire_display_nm.stl").exists():
            print("目录中的旧 STL 可能对应之前的模型；请运行第二步更新。")
    elif STEP == "stl":
        stl_path = export_cached_display_stl(config)
        print(f"展示用 STL：{stl_path}")
    else:
        raise ValueError('STEP 只能填写 "image" 或 "stl"')


if __name__ == "__main__":
    main()
