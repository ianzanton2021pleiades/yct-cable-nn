"""Real labeled FDR cases with known joint/end positions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\FDR案例-csv")
RG58_FACTOR = 172.0 / 198.0  # model distance axis uses 172 m/us; RG58 vf ~= 0.66


@dataclass(frozen=True)
class RealCase:
    path: Path
    events_m: tuple[float, ...]
    end_m: float
    group: str
    note: str = ""


def read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
    names = data.dtype.names
    return (np.asarray(data[names[0]], float),
            np.asarray(data[names[1]], float) + 1j * np.asarray(data[names[2]], float))


def wet_cases() -> list[RealCase]:
    folder = ROOT / "1500m电缆单个接头浸水试验"
    cases = []
    for pos in (100, 200, 300, 400, 500, 600, 700, 900, 1100, 1300, 1400):
        path = folder / f"1500m电缆单个接头浸水试验-{pos}m接头浸水.csv"
        cases.append(RealCase(path, (float(pos),), 1500.0, "wet_1500", f"joint_{pos}"))
    return cases


def rg58_74_cases() -> list[RealCase]:
    folder = ROOT / "RG58-74M(40+4+30)"
    cases = []
    for path in sorted(folder.glob("*.csv")):
        cases.append(RealCase(path, (40.0 * RG58_FACTOR, 44.0 * RG58_FACTOR), 74.0 * RG58_FACTOR,
                              "rg58_74", path.stem))
    return cases


def rg58_3line_cases() -> list[RealCase]:
    folder = ROOT / "RG58-3Lines"
    cases = []
    for path in sorted(folder.glob("*.csv")):
        cases.append(RealCase(path, (40.0 * RG58_FACTOR, 41.0 * RG58_FACTOR, 66.0 * RG58_FACTOR),
                              96.0 * RG58_FACTOR, "rg58_3line", path.stem))
    return cases


def all_cases() -> list[RealCase]:
    return wet_cases() + rg58_74_cases() + rg58_3line_cases()
