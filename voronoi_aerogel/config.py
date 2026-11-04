from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class ModelConfig:
    """用户可调的模型参数。

    所有长度均以纳米存储，几何在 x、y、z 三个方向周期平铺。
    ``target_porosity`` 使用 0--1 小数，例如 0.95 表示 95% 孔隙率。
    """

    cube_size_nm: float = 1000.0
    thickness_nm: float = 5.0
    diameter_min_nm: float = 50.0
    diameter_max_nm: float = 80.0
    target_porosity: float = 0.95
    porosity_tolerance: float = 0.005
    min_in_range_fraction: float = 0.90
    random_seed: int = 42
    num_seeds: int | None = None
    seed_min_distance_factor: float = 0.32
    qmc_power: int = 20
    validation_qmc_power: int = 19
    voxel_size_nm: float = 2.5
    max_generation_attempts: int = 4
    max_selected_faces: int = 25_000
    min_face_area_nm2: float = 25.0
    min_face_edge_nm: float = 1.0
    winding_offset_bound: int = 1
    preview_face_limit: int = 15_000
    output_dir: str = "output/default_model"

    @property
    def target_diameter_nm(self) -> float:
        """用户片径窗口的中心值，用于估算母点数量。"""

        return 0.5 * (self.diameter_min_nm + self.diameter_max_nm)

    @property
    def target_solid_fraction(self) -> float:
        """目标固相体积分数，即 1-孔隙率。"""

        return 1.0 - self.target_porosity

    def validate(self) -> None:
        """在执行昂贵几何运算前检查用户输入。"""

        if self.cube_size_nm <= 0:
            raise ValueError("cube_size_nm must be positive")
        if self.thickness_nm <= 0:
            raise ValueError("thickness_nm must be positive")
        if not 0.0 < self.target_porosity < 1.0:
            raise ValueError("target_porosity must lie strictly between 0 and 1")
        if not 0.0 <= self.porosity_tolerance < 1.0:
            raise ValueError("porosity_tolerance must lie in [0, 1)")
        if not 0.0 < self.diameter_min_nm < self.diameter_max_nm:
            raise ValueError("diameter range must satisfy 0 < min < max")
        if not 0.0 <= self.min_in_range_fraction <= 1.0:
            raise ValueError("min_in_range_fraction must lie in [0, 1]")
        if not 12 <= self.qmc_power <= 24:
            raise ValueError("qmc_power must lie in [12, 24]")
        if not 12 <= self.validation_qmc_power <= 24:
            raise ValueError("validation_qmc_power must lie in [12, 24]")
        if self.voxel_size_nm <= 0:
            raise ValueError("voxel_size_nm must be positive")
        if self.max_selected_faces <= 0:
            raise ValueError("max_selected_faces must be positive")

    def to_dict(self) -> dict[str, object]:
        """返回可直接写入 JSON 的配置字典。"""

        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """从 JSON 文件读取并验证配置。"""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(**data)
        config.validate()
        return config

    def write_json(self, path: str | Path) -> None:
        """以 UTF-8 格式保存实际使用的参数快照。"""

        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
