"""Reanalyse COMSOL and measured RG58/RG316 references for the DG route.

The source archives are read in place.  No archive is extracted or modified.
Outputs are deliberately kept under this experiment's ignored ``output`` and
``assets`` directories; the script and the generated report remain reviewable.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import re
import sys
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFERENCE_ROOT = ROOT / "REF" / "20260903线路仿真实测数据分析"
V1_PATH = HERE.parent / "CST_Reproduction" / "V1" / "cst_fdr_reproduction.py"
NOMINAL_TOTAL_LENGTH_M = 71.0
ZREF_OHM = 50.0
C0 = 299_792_458.0


def load_v1():
    spec = spec_from_file_location("reference_reanalysis_v1", V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {V1_PATH}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V1 = load_v1()


def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def safe_name(value: str) -> str:
    return str(value).encode("unicode_escape").decode("ascii")


def select_member(zpath: Path, predicate) -> str:
    with zipfile.ZipFile(zpath) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    matches = [name for name in names if predicate(name)]
    if len(matches) != 1:
        raise RuntimeError(
            f"{zpath.name}: expected one CSV, got {len(matches)}: "
            + ", ".join(safe_name(item) for item in matches)
        )
    return matches[0]


def read_member(zpath: Path, member: str) -> bytes:
    with zipfile.ZipFile(zpath) as archive:
        return archive.read(member)


def read_s11_csv(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(io.StringIO(decode_text(raw)))
    frame = frame.dropna(axis=1, how="all")
    normalized = {
        str(column).lower().replace("_", "").replace(" ", ""): column
        for column in frame.columns
    }
    frequency_column = next(column for key, column in normalized.items() if "freq" in key)
    real_column = next(column for key, column in normalized.items() if "real" in key)
    imaginary_column = next(column for key, column in normalized.items() if "imag" in key)
    frequency = pd.to_numeric(frame[frequency_column], errors="coerce").to_numpy(float)
    s11 = pd.to_numeric(frame[real_column], errors="coerce").to_numpy(float)
    s11 = s11 + 1j * pd.to_numeric(frame[imaginary_column], errors="coerce").to_numpy(float)
    keep = np.isfinite(frequency) & np.isfinite(s11.real) & np.isfinite(s11.imag)
    frequency = frequency[keep]
    s11 = s11[keep]
    order = np.argsort(frequency, kind="stable")
    frequency = frequency[order]
    s11 = s11[order]
    unique, indices = np.unique(frequency, return_index=True)
    return unique, s11[indices]


def read_s11_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return read_s11_csv(path.read_bytes())


def parse_complex(value: object) -> complex:
    text = str(value).strip().replace("i", "j").replace("I", "j")
    if text.lower() in {"nan", "", "none"}:
        return complex(np.nan, np.nan)
    return complex(text)


def read_comsol_csv(raw: bytes) -> pd.DataFrame:
    lines = decode_text(raw).splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.lstrip().startswith("% freq"))
    data = pd.read_csv(io.StringIO("\n".join(lines[header_index + 1 :])), header=None)
    if data.shape[1] < 7:
        raise RuntimeError(f"COMSOL table has {data.shape[1]} columns, expected at least 7")
    columns = ["frequency_hz", "R_ohm_per_m", "L_h_per_m", "G_s_per_m", "C_f_per_m", "Zc_ohm", "gamma_per_m"]
    output = pd.DataFrame()
    for index, column in enumerate(columns[:5]):
        output[column] = pd.to_numeric(data.iloc[:, index], errors="coerce")
    output["Zc_ohm"] = data.iloc[:, 5].map(parse_complex)
    output["gamma_per_m"] = data.iloc[:, 6].map(parse_complex)
    return output


def read_comsol_archive(zpath: Path) -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(zpath) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        result = {}
        for name in sorted(names):
            result[Path(name).stem] = read_comsol_csv(archive.read(name))
    return result


def temperature_from_member(name: str) -> int:
    match = re.search(r"-(\d+)[^0-9]*C", name)
    if match is None:
        raise RuntimeError(f"cannot parse temperature from {safe_name(name)}")
    return int(match.group(1))


def read_temperature_archive(zpath: Path) -> dict[int, tuple[str, np.ndarray, np.ndarray]]:
    with zipfile.ZipFile(zpath) as archive:
        result = {}
        for name in archive.namelist():
            if name.lower().endswith(".csv"):
                temperature = temperature_from_member(name)
                result[temperature] = (name, *read_s11_csv(archive.read(name)))
    return result


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"q05": math.nan, "q50": math.nan, "q95": math.nan}
    return {"q05": float(np.quantile(values, 0.05)), "q50": float(np.quantile(values, 0.50)), "q95": float(np.quantile(values, 0.95))}


def band_masks(frequency_hz: np.ndarray):
    bands = (
        ("9k_100k", 9.0e3, 1.0e5),
        ("100k_1M", 1.0e5, 1.0e6),
        ("1M_10M", 1.0e6, 1.0e7),
        ("10M_100M", 1.0e7, 1.0e8),
        ("100M_500M", 1.0e8, 5.0e8),
        ("500M_1G", 5.0e8, 1.0e9),
        ("1G_2G", 1.0e9, 2.1e9),
    )
    for name, low, high in bands:
        yield name, (frequency_hz >= low) & (frequency_hz < high)


def s11_summary(frequency_hz: np.ndarray, s11: np.ndarray) -> dict[str, object]:
    magnitude = np.abs(s11)
    differences = np.diff(frequency_hz)
    result: dict[str, object] = {
        "points": int(frequency_hz.size),
        "frequency_start_hz": float(frequency_hz[0]),
        "frequency_stop_hz": float(frequency_hz[-1]),
        "frequency_step_median_hz": float(np.median(differences)) if differences.size else math.nan,
        "frequency_step_min_hz": float(np.min(differences)) if differences.size else math.nan,
        "frequency_step_max_hz": float(np.max(differences)) if differences.size else math.nan,
        "first_point": {"real": float(s11[0].real), "imag": float(s11[0].imag), "magnitude": float(magnitude[0])},
        "magnitude": {**quantiles(magnitude), "max": float(np.max(magnitude)), "fraction_gt_1": float(np.mean(magnitude > 1.0))},
        "bands": {},
    }
    for name, mask in band_masks(frequency_hz):
        if np.any(mask):
            result["bands"][name] = {**quantiles(magnitude[mask]), "max": float(np.max(magnitude[mask])), "fraction_gt_1": float(np.mean(magnitude[mask] > 1.0))}
    return result


def continuous_sqrt(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, complex)
    result = np.sqrt(values)
    for index in range(1, result.size):
        candidate = result[index]
        if abs(-candidate - result[index - 1]) < abs(candidate - result[index - 1]):
            result[index] = -candidate
    return result


def recover_three_load_abcd(
    frequency_hz: np.ndarray,
    open_s11: np.ndarray,
    short_s11: np.ndarray,
    matched_s11: np.ndarray,
    total_length_m: float = NOMINAL_TOTAL_LENGTH_M,
) -> dict[str, np.ndarray]:
    """Recover a reciprocal ABCD matrix from three same-plane one-port loads.

    This is a diagnostic equivalent network.  The cable contains joints and a
    1 m section, so the derived RLGC is not treated as a direct homogeneous
    RG58 material measurement.
    """
    def z_in(gamma: np.ndarray) -> np.ndarray:
        return ZREF_OHM * (1.0 + gamma) / (1.0 - gamma)

    z_open = z_in(open_s11)
    z_short = z_in(short_s11)
    z_matched = z_in(matched_s11)
    q = (z_short - z_matched) / (ZREF_OHM * (z_matched - z_open))
    denominator = q * (z_open - z_short)
    d = continuous_sqrt(1.0 / denominator)
    c = q * d
    a = z_open * c
    b = z_short * d
    zc = continuous_sqrt(b / c)
    trace_half = (a + d) / 2.0
    gamma_l = np.arccosh(trace_half)
    gamma_l = gamma_l.real + 1j * np.unwrap(gamma_l.imag)
    gamma_per_m = gamma_l / float(total_length_m)
    z_series = gamma_per_m * zc
    y_shunt = gamma_per_m / zc
    omega = 2.0 * np.pi * frequency_hz
    return {
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "Z_open_ohm": z_open,
        "Z_short_ohm": z_short,
        "Z_matched_ohm": z_matched,
        "Zc_ohm": zc,
        "gamma_per_m": gamma_per_m,
        "R_ohm_per_m": z_series.real,
        "L_h_per_m": z_series.imag / omega,
        "G_s_per_m": y_shunt.real,
        "C_f_per_m": y_shunt.imag / omega,
        "relative_A_D_difference": np.abs(a - d) / np.maximum(1.0, np.abs(a) + np.abs(d)),
        "valid": np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d),
    }


def effective_rlgc_summary(
    frequency_hz: np.ndarray,
    recovered: dict[str, np.ndarray],
    total_length_m: float = NOMINAL_TOTAL_LENGTH_M,
) -> dict[str, object]:
    output: dict[str, object] = {"nominal_total_length_m": float(total_length_m), "bands": {}}
    for name, mask in band_masks(frequency_hz):
        mask = mask & recovered["valid"]
        if not np.any(mask):
            continue
        output["bands"][name] = {
            "points": int(np.sum(mask)),
            "R_ohm_per_m": quantiles(recovered["R_ohm_per_m"][mask]),
            "L_nh_per_m": quantiles(recovered["L_h_per_m"][mask] * 1.0e9),
            "G_us_per_m": quantiles(recovered["G_s_per_m"][mask] * 1.0e6),
            "C_pf_per_m": quantiles(recovered["C_f_per_m"][mask] * 1.0e12),
            "Zc_abs_ohm": quantiles(np.abs(recovered["Zc_ohm"][mask])),
            "relative_A_D_difference": quantiles(recovered["relative_A_D_difference"][mask]),
            "positive_R_fraction": float(np.mean(recovered["R_ohm_per_m"][mask] >= 0.0)),
            "positive_L_fraction": float(np.mean(recovered["L_h_per_m"][mask] >= 0.0)),
            "positive_G_fraction": float(np.mean(recovered["G_s_per_m"][mask] >= 0.0)),
            "positive_C_fraction": float(np.mean(recovered["C_f_per_m"][mask] >= 0.0)),
        }
    return output


def run_fdr(frequency_hz: np.ndarray, s11: np.ndarray, time_points: int) -> tuple[dict, dict]:
    a1 = V1.algorithm1_ifft(frequency_hz, s11)
    with contextlib.redirect_stdout(io.StringIO()):
        a2 = V1.algorithm2_fdr(
            frequency_hz,
            s11,
            time_points=time_points,
            chunk_size=256,
            progress_label="",
        )
    return a1, a2


def peak_in_window(distance: np.ndarray, values: np.ndarray, low: float, high: float) -> dict[str, float]:
    mask = (distance >= low) & (distance <= high) & np.isfinite(values)
    if not np.any(mask):
        return {"position_m": math.nan, "value": math.nan, "abs_value": math.nan}
    indices = np.flatnonzero(mask)
    index = indices[int(np.argmax(np.abs(values[indices])))]
    return {"position_m": float(distance[index]), "value": float(values[index]), "abs_value": float(abs(values[index]))}


def fdr_summary_for_length(a1: dict, a2: dict, length_m: float) -> dict[str, object]:
    a1_impulse = np.asarray(a1["impulse_real_ref"])
    a2_impulse = np.asarray(a2["impulse_smoothed"])
    a1_step = np.asarray(a1["step_ref"])
    a2_step = np.asarray(a2["step_smoothed"])
    low = max(0.5 * float(length_m), 1.0)
    high = max(1.5 * float(length_m), low + 1.0)
    return {
        "algorithm1": {
            "first_step_hz": float(a1["first_step_hz"]),
            "distance_max_m": float(a1["distance_m"][-1]),
            "terminal_impulse_window": peak_in_window(np.asarray(a1["distance_m"]), a1_impulse, low, high),
            "near_impulse_0_8m": peak_in_window(np.asarray(a1["distance_m"]), a1_impulse, 0.0, 8.0),
            "terminal_step_window": peak_in_window(np.asarray(a1["distance_m"]), a1_step, low, high),
        },
        "algorithm2": {
            "distance_max_m": float(a2["distance_m"][-1]),
            "terminal_impulse_window": peak_in_window(np.asarray(a2["distance_impulse_m"]), a2_impulse, low, high),
            "near_impulse_0_8m": peak_in_window(np.asarray(a2["distance_impulse_m"]), a2_impulse, 0.0, 8.0),
            "terminal_step_window": peak_in_window(np.asarray(a2["distance_m"]), a2_step, low, high),
        },
    }


def fdr_summary(a1: dict, a2: dict) -> dict[str, object]:
    return fdr_summary_for_length(a1, a2, NOMINAL_TOTAL_LENGTH_M)


def configure_plot() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def save_s11_plot(path: Path, traces: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), dpi=200)
    for label, (frequency_hz, s11) in traces.items():
        x = frequency_hz / 1.0e6
        axes[0, 0].plot(x, 20.0 * np.log10(np.maximum(np.abs(s11), 1.0e-12)), lw=0.75, label=label)
        axes[0, 1].plot(x, np.angle(s11, deg=True), lw=0.65, label=label)
        axes[1, 0].plot(x, s11.real, lw=0.65, label=label)
        axes[1, 1].plot(x, s11.imag, lw=0.65, label=label)
    titles = ("S11 幅值", "S11 包裹相位", "S11 实部", "S11 虚部")
    ylabels = ("dB", "deg", "Real", "Imag")
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set_title(title)
        axis.set_xlabel("Frequency (MHz)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, min(1_000.0, max(float(f[-1] / 1.0e6) for f, _ in traces.values())))
        axis.grid(True, lw=0.4, color="#cbd5e1")
        axis.tick_params(direction="in", top=True, right=True)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_fdr_plot(path: Path, traces: dict[str, tuple[dict, dict]]) -> None:
    configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), dpi=200, sharex=False)
    for label, (a1, a2) in traces.items():
        axes[0, 0].plot(a1["distance_m"], a1["impulse_real_ref"], lw=0.7, label=label)
        axes[0, 1].plot(a1["distance_m"], a1["step_ref"], lw=0.7, label=label)
        axes[1, 0].plot(a2["distance_impulse_m"], a2["impulse_smoothed"], lw=0.7, label=label)
        axes[1, 1].plot(a2["distance_m"], a2["step_smoothed"], lw=0.7, label=label)
    titles = ("算法1 脉冲", "算法1 阶跃", "算法2 脉冲", "算法2 阶跃")
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("Distance (m)")
        axis.grid(True, lw=0.4, color="#cbd5e1")
        axis.tick_params(direction="in", top=True, right=True)
        axis.set_xlim(0.0, 95.0)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_comsol_plot(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), dpi=200)
    styles = {"R_ohm_per_m": ("R (Ω/m)", "R"), "L_h_per_m": ("L (nH/m)", "L"), "G_s_per_m": ("G (S/m)", "G"), "C_f_per_m": ("C (pF/m)", "C")}
    for stem, table in tables.items():
        frequency = table["frequency_hz"].to_numpy(float)
        positive = frequency > 0.0
        for axis, column in zip(axes.flat, styles):
            scale = 1.0e9 if column == "L_h_per_m" else 1.0e12 if column == "C_f_per_m" else 1.0
            axis.plot(frequency[positive] / 1.0e6, table.loc[positive, column].to_numpy(float) * scale, lw=0.8, label=stem)
    for axis, column in zip(axes.flat, styles):
        axis.set_title(styles[column][0])
        axis.set_xlabel("Frequency (MHz)")
        axis.set_xscale("log")
        axis.grid(True, lw=0.4, color="#cbd5e1")
        axis.tick_params(direction="in", top=True, right=True)
        axis.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def comsol_table_summary(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    sample_frequencies = (9.0e3, 1.0e5, 1.0e6, 1.0e7, 1.0e8, 2.0e8)
    output: dict[str, object] = {}
    for stem, table in tables.items():
        frequency = table["frequency_hz"].to_numpy(float)
        rows = []
        for target in sample_frequencies:
            index = int(np.nanargmin(np.abs(frequency - target)))
            row = {"requested_hz": target, "frequency_hz": float(frequency[index])}
            for column, scale in (("R_ohm_per_m", 1.0), ("L_h_per_m", 1.0e9), ("G_s_per_m", 1.0e12), ("C_f_per_m", 1.0e12)):
                value = table[column].iloc[index]
                row[column] = float(value * scale) if np.isfinite(value) else math.nan
            zc = table.iloc[index]["Zc_ohm"]
            gamma = table.iloc[index]["gamma_per_m"]
            row["Zc_real_ohm"] = float(zc.real) if np.isfinite(zc.real) else math.nan
            row["Zc_imag_ohm"] = float(zc.imag) if np.isfinite(zc.imag) else math.nan
            row["alpha_np_per_m"] = float(gamma.real) if np.isfinite(gamma.real) else math.nan
            row["beta_rad_per_m"] = float(gamma.imag) if np.isfinite(gamma.imag) else math.nan
            rows.append(row)
        output[stem] = {
            "points": int(len(table)),
            "frequency_start_hz": float(table["frequency_hz"].iloc[0]),
            "frequency_stop_hz": float(table["frequency_hz"].iloc[-1]),
            "rows": rows,
        }
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "output" / "reference_reanalysis")
    parser.add_argument("--assets", type=Path, default=HERE / "assets" / "reference_reanalysis")
    parser.add_argument("--time-points", type=int, default=10_000)
    args = parser.parse_args()
    output = args.output.resolve()
    assets = args.assets.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    comsol_zip = REFERENCE_ROOT / "COMSOL RG58 RLGC仿真结果.zip"
    terminal_zip = REFERENCE_ROOT / "RG58 40m-1m-30m不同末端.zip"
    temperature_zip = REFERENCE_ROOT / "RG58-RG316.zip"
    direct_csv = REFERENCE_ROOT / "Core-LineA+CUT1+LineB(20degree)-1.csv"

    comsol_tables = read_comsol_archive(comsol_zip)
    terminal_traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    terminal_members: dict[str, str] = {}
    with zipfile.ZipFile(terminal_zip) as archive:
        for label, token in (("open", "-open"), ("short", "-short"), ("50ohm", "-50ohm")):
            member = next(name for name in archive.namelist() if token in name and name.lower().endswith(".csv"))
            terminal_members[label] = member
            terminal_traces[label] = read_s11_csv(archive.read(member))

    temperature_data = read_temperature_archive(temperature_zip)
    direct_frequency, direct_s11 = read_s11_file(direct_csv)

    terminal_summaries = {label: s11_summary(*trace) for label, trace in terminal_traces.items()}
    terminal_fdr: dict[str, tuple[dict, dict]] = {}
    terminal_fdr_summary = {}
    for label, (frequency, s11) in terminal_traces.items():
        a1, a2 = run_fdr(frequency, s11, args.time_points)
        terminal_fdr[label] = (a1, a2)
        terminal_fdr_summary[label] = fdr_summary(a1, a2)
        np.savez_compressed(output / f"terminal_{label}_fdr.npz", frequency_hz=frequency, s11=s11, algorithm1_distance_m=a1["distance_m"], algorithm1_impulse_raw=a1["impulse_real_raw"], algorithm1_impulse=a1["impulse_real_ref"], algorithm1_step_raw=a1["step_raw"], algorithm1_step=a1["step_ref"], algorithm2_distance_m=a2["distance_m"], algorithm2_impulse_distance_m=a2["distance_impulse_m"], algorithm2_impulse_raw=a2["impulse_raw"], algorithm2_impulse=a2["impulse_smoothed"], algorithm2_step_raw=a2["step_raw"], algorithm2_step=a2["step_smoothed"])

    # The first recorded 9 kHz point is a known instrument-save artifact.  It
    # is retained in the source inventory but excluded from the inverse model.
    recovered_frequency = terminal_traces["open"][0][1:]
    recovered = recover_three_load_abcd(
        recovered_frequency,
        terminal_traces["open"][1][1:],
        terminal_traces["short"][1][1:],
        terminal_traces["50ohm"][1][1:],
    )
    effective_summary = effective_rlgc_summary(recovered_frequency, recovered)
    effective_rows = []
    frequency = recovered_frequency
    for index, value in enumerate(frequency):
        effective_rows.append({
            "Frequency_Hz": float(value),
            "R_ohm_per_m": float(recovered["R_ohm_per_m"][index]),
            "L_nH_per_m": float(recovered["L_h_per_m"][index] * 1.0e9),
            "G_uS_per_m": float(recovered["G_s_per_m"][index] * 1.0e6),
            "C_pF_per_m": float(recovered["C_f_per_m"][index] * 1.0e12),
            "Zc_abs_ohm": float(abs(recovered["Zc_ohm"][index])),
            "A_D_relative_difference": float(recovered["relative_A_D_difference"][index]),
        })
    write_csv(output / "three_load_effective_rlgc.csv", effective_rows)

    temperature_summaries = {}
    temperature_fdr_summary = {}
    temperature_fdr: dict[int, tuple[dict, dict]] = {}
    reference_temperature = min(temperature_data)
    for temperature, (_, frequency_t, s11_t) in sorted(temperature_data.items()):
        temperature_summaries[str(temperature)] = s11_summary(frequency_t, s11_t)
        a1, a2 = run_fdr(frequency_t, s11_t, args.time_points)
        temperature_fdr[temperature] = (a1, a2)
        temperature_fdr_summary[str(temperature)] = fdr_summary(a1, a2)
        np.savez_compressed(output / f"rg316_{temperature}C_fdr.npz", frequency_hz=frequency_t, s11=s11_t, algorithm1_distance_m=a1["distance_m"], algorithm1_impulse_raw=a1["impulse_real_raw"], algorithm1_impulse=a1["impulse_real_ref"], algorithm1_step_raw=a1["step_raw"], algorithm1_step=a1["step_ref"], algorithm2_distance_m=a2["distance_m"], algorithm2_impulse_distance_m=a2["distance_impulse_m"], algorithm2_impulse_raw=a2["impulse_raw"], algorithm2_impulse=a2["impulse_smoothed"], algorithm2_step_raw=a2["step_raw"], algorithm2_step=a2["step_smoothed"])

    direct_a1, direct_a2 = run_fdr(direct_frequency, direct_s11, args.time_points)
    direct_summary = {"s11": s11_summary(direct_frequency, direct_s11), "fdr": fdr_summary(direct_a1, direct_a2)}

    save_s11_plot(assets / "rg58_three_terminations_s11.png", terminal_traces)
    save_fdr_plot(assets / "rg58_three_terminations_fdr.png", terminal_fdr)
    save_comsol_plot(assets / "comsol_rg58_rlgc.png", comsol_tables)

    payload = {
        "protocol": {
            "source_root": str(REFERENCE_ROOT),
            "terminal_members": terminal_members,
            "temperature_members": {str(key): value[0] for key, value in temperature_data.items()},
            "nominal_topology": "instrument -> SMA-BNC -> 40 m RG58 -> BNC -> 1 m RG58 -> BNC -> 30 m RG58 -> termination",
            "nominal_total_length_m": NOMINAL_TOTAL_LENGTH_M,
            "fdr_time_points": args.time_points,
            "algorithm1_and_algorithm2": "current CST_Reproduction/V1 implementation; algorithm1 uses complex S11, algorithm2 uses S11 real",
            "comsol_column_mapping": "frequency, R, L, G, C, Zc, gamma from the exported transmission-line-parameter table",
        },
        "comsol": comsol_table_summary(comsol_tables),
        "terminal_s11": terminal_summaries,
        "terminal_fdr": terminal_fdr_summary,
        "three_load_effective_rlgc": effective_summary,
        "temperature_s11": temperature_summaries,
        "temperature_fdr": temperature_fdr_summary,
        "direct_rg58_reference": direct_summary,
        "notes": [
            "The 9 kHz point is retained in inventory but algorithm1/2 skip the first point under the current V1 protocol.",
            "Three-load RLGC is an effective diagnostic of the complete reciprocal two-port under a nominal 71 m homogeneous-line assumption; it is not a direct de-embedded RG58 material measurement because the topology has BNC joints and a 1 m section.",
            "Raw measured S11 is not forcibly clipped to unit magnitude; values above one are reported as a measurement/fixture quality issue.",
        ],
    }
    (output / "reference_reanalysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "assets": str(assets),
        "terminal_points": {key: value["points"] for key, value in terminal_summaries.items()},
        "temperature_points": {key: value["points"] for key, value in temperature_summaries.items()},
        "comsol_tables": list(comsol_tables),
        "direct_points": direct_summary["s11"]["points"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
