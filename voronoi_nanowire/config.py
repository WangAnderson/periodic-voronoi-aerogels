from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class NanowireConfig:
    """Configuration for the periodic Voronoi nanowire display model.

    Lengths are expressed in nanometres.  ``target_porosity`` and
    ``porosity_tolerance`` are fractions, not percentages.
    """

    cube_size_nm: float = 1000.0
    num_seeds: int = 400
    seed_min_distance_factor: float = 0.30
    rod_diameter_nm: float = 20.0
    target_porosity: float = 0.95
    porosity_tolerance: float = 0.005
    random_seed: int = 42
    qmc_power: int = 18
    validation_qmc_power: int = 19
    tube_sides: int = 12
    output_dir: str = "output/nanowire_default"

    @property
    def rod_radius_nm(self) -> float:
        return 0.5 * self.rod_diameter_nm

    @property
    def target_solid_fraction(self) -> float:
        return 1.0 - self.target_porosity

    def validate(self) -> None:
        if self.cube_size_nm <= 0.0:
            raise ValueError("cube_size_nm must be positive")
        if self.num_seeds < 5:
            raise ValueError("num_seeds must be at least 5")
        if not 0.0 <= self.seed_min_distance_factor < 1.0:
            raise ValueError("seed_min_distance_factor must lie in [0, 1)")
        if not 0.0 < self.rod_diameter_nm < self.cube_size_nm:
            raise ValueError("rod_diameter_nm must lie in (0, cube_size_nm)")
        if not 0.0 < self.target_porosity < 1.0:
            raise ValueError("target_porosity must lie strictly between 0 and 1")
        if not 0.0 <= self.porosity_tolerance < 1.0:
            raise ValueError("porosity_tolerance must lie in [0, 1)")
        if not 12 <= self.qmc_power <= 24:
            raise ValueError("qmc_power must lie in [12, 24]")
        if not 12 <= self.validation_qmc_power <= 24:
            raise ValueError("validation_qmc_power must lie in [12, 24]")
        if self.tube_sides < 6:
            raise ValueError("tube_sides must be at least 6")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "NanowireConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(**data)
        config.validate()
        return config

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


__all__ = ["NanowireConfig"]
