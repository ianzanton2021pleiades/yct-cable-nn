# -*- coding: utf-8 -*-
"""
FDR数据文件快速转CSV —— V2 极速版（纯 Python，无 DLL 依赖）
================================================================

.fdr/.db8 文件原理（本版本已完全逆向确认）：
  * 本质是一个 SQLite 3 数据库（厂商 exe 内嵌 SQLite 3.14.2，编译了 HAS_CODEC 加密层）
  * 加密方式为 System.Data.SQLite 的 legacy codec：
      - 密钥 = SHA-1(密码 "CZY") 的前 16 字节
      - RC4 流密码，且 keystream 按页（默认 4096 字节）重复使用
      - 整个文件（包括页首 16 字节的 SQLite 标志头）全部被加密
  * 因此解密 = 生成一次 4096 字节 keystream，与全文件按页 XOR，即可得到明文 SQLite 库

与旧版相比：
  * 不再需要 pythonnet / System.Data.SQLite.dll / SQLite.Interop.dll
  * 解密用 numpy 按页 XOR（无 numpy 时自动退化为纯标准库大整数 XOR）
  * 数据库用标准库 sqlite3 读取（Python 3.11+ 内存反序列化，否则落临时文件）
  * "不应用校准"模式输出与旧版工具逐字节一致；校准模式公式与旧版完全相同
  * 速度快 1~2 个数量级（8MB 样例：旧版数秒~半分钟 → 新版约 0.3s）

命令行批量：python 本文件 文件1.fdr 文件2.db8 ...（输出到各自源目录，不校准）
不带参数则启动图形界面。
"""
from __future__ import annotations

import hashlib
import math
import os
import queue
import re
import sqlite3
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as _np
except ImportError:  # 纯标准库也能跑，只是解密/校准慢一点
    _np = None


PASSWORD = "CZY"
TITLE = "FDR数据文件快速转CSV"
FILETYPES = [("工程数据文件", "*.fdr *.db8"), ("All files", "*.*")]
INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')
Z0 = 50.0
GAMMA_EPS = 1e-15

CAL_MODE_NONE = "不应用校准"
CAL_MODE_REF_NORM = "参考归一化（错误）"
CAL_MODE_DELAY_PHASE = "时延相位补偿"
CAL_MODE_SERIES = "去串联寄生"
CALIBRATION_OPTIONS = [
    CAL_MODE_NONE,
    CAL_MODE_REF_NORM,
    CAL_MODE_DELAY_PHASE,
    CAL_MODE_SERIES,
]


# ---------------------------------------------------------------------------
# 加密与解密
# ---------------------------------------------------------------------------

def rc4_keystream(key: bytes, n: int) -> bytes:
    """RC4 PRGA，输出 n 字节 keystream。"""
    s = list(range(256))
    j = 0
    klen = len(key)
    for i in range(256):
        j = (j + s[i] + key[i % klen]) & 0xFF
        s[i], s[j] = s[j], s[i]
    out = bytearray(n)
    i = j = 0
    for pos in range(n):
        i = (i + 1) & 0xFF
        j = (j + s[i]) & 0xFF
        s[i], s[j] = s[j], s[i]
        out[pos] = s[(s[i] + s[j]) & 0xFF]
    return bytes(out)


class FdrCrypto:
    """System.Data.SQLite legacy codec 解密器。

    key = SHA-1(密码)[:16]；RC4 keystream 每 page_size 字节重复一遍。
    """

    PAGE_CANDIDATES = (4096, 8192, 1024, 2048, 512, 16384, 32768, 65536)
    MAGIC = b"SQLite format 3\x00"

    def __init__(self, password: str = PASSWORD) -> None:
        self.key = hashlib.sha1(password.encode("utf-8")).digest()[:16]
        self._ks_cache: dict[int, bytes] = {}

    def _keystream(self, page_size: int) -> bytes:
        ks = self._ks_cache.get(page_size)
        if ks is None:
            ks = rc4_keystream(self.key, page_size)
            self._ks_cache[page_size] = ks
        return ks

    def decrypt(self, data: bytes) -> tuple[bytes, int]:
        """返回 (明文SQLite字节串, 页大小)。"""
        if len(data) < 4096:
            raise RuntimeError("文件太小，不是有效的 .fdr/.db8 数据库")

        page_size = 0
        head = b""
        for ps in self.PAGE_CANDIDATES:
            ks = self._keystream(ps)
            head = bytes(a ^ b for a, b in zip(data[:ps], ks))
            if head[:16] == self.MAGIC:
                page_size = ps
                break
        if not page_size:
            raise RuntimeError("无法解密文件：不是加密的 .fdr/.db8，或密码不正确")

        # 从解密出的 SQLite 头里核实真实页大小
        field_value = struct.unpack(">H", head[16:18])[0]
        real_page = 65536 if field_value == 1 else field_value
        if real_page > 0 and len(data) % real_page == 0:
            page_size = real_page

        ks = self._keystream(page_size)
        plain = self._xor_repeated(data, ks)
        return plain, page_size

    @staticmethod
    def _xor_repeated(data: bytes, ks: bytes) -> bytes:
        """data 与循环使用的 keystream 异或。"""
        ps = len(ks)
        if _np is not None:
            n = (len(data) // ps) * ps
            main = _np.frombuffer(data[:n], dtype=_np.uint8)
            plain = _np.bitwise_xor(main.reshape(-1, ps), _np.frombuffer(ks, dtype=_np.uint8)).tobytes()
            tail = data[n:]
            if tail:
                plain += bytes(a ^ b for a, b in zip(tail, ks))
            return plain
        # 纯标准库：大整数 XOR（CPython 的 int 异或是 C 实现，速度可接受）
        n = len(data)
        reps = -(-n // ps)
        ks_full = (ks * reps)[:n]
        return (int.from_bytes(data, "big") ^ int.from_bytes(ks_full, "big")).to_bytes(n, "big")


def _open_sqlite(plain: bytes) -> tuple[sqlite3.Connection, Path | None]:
    """打开解密后的 SQLite 字节串：优先内存反序列化（Py3.11+），否则临时文件。"""
    con = sqlite3.connect(":memory:")
    deserialize = getattr(con, "deserialize", None)
    if deserialize is not None:
        try:
            deserialize(plain)
            return con, None
        except sqlite3.Error:
            pass
    con.close()
    fd, tmp_name = tempfile.mkstemp(prefix="fdr_v2_", suffix=".db")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(plain)
        return sqlite3.connect(tmp_name), Path(tmp_name)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


class FdrDatabase:
    """解密并打开一个 .fdr/.db8。"""

    def __init__(self, db_path: str | os.PathLike[str], crypto: FdrCrypto | None = None) -> None:
        self.crypto = crypto or FdrCrypto()
        t0 = time.perf_counter()
        data = Path(db_path).read_bytes()
        self.plain, self.page_size = self.crypto.decrypt(data)
        t1 = time.perf_counter()
        self.con, self._tmp_path = _open_sqlite(self.plain)
        t2 = time.perf_counter()
        self.decrypt_seconds = t1 - t0
        self.open_seconds = t2 - t1
        if self._tmp_path is None:
            try:
                self.con.execute("PRAGMA integrity_check")
            except sqlite3.Error as exc:
                raise RuntimeError(f"解密后的数据库校验失败：{exc}") from exc

    def tables(self) -> set[str]:
        rows = self.con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return {str(r[0]) for r in rows if r[0] is not None}

    def rows(self, sql: str) -> list[tuple]:
        return self.con.execute(sql).fetchall()

    def close(self) -> None:
        try:
            self.con.close()
        finally:
            if self._tmp_path is not None:
                try:
                    os.remove(self._tmp_path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# 导出（输出格式与旧版完全一致）
# ---------------------------------------------------------------------------

@dataclass
class ExportItem:
    display_name: str
    table_name: str
    is_calibration: bool = False


@dataclass
class CalibrationContext:
    mode: str
    calibration: list[tuple[float, complex]]
    tau: float | None = None
    series_r: float | None = None
    series_l: float | None = None
    freqs: Any = None  # numpy 数组（REF_NORM 用）
    xs: Any = None
    reals: Any = None
    imags: Any = None
    interpolator: "ComplexLogInterpolator | None" = None  # 纯 Python 回退用


class ComplexLogInterpolator:
    """log10(频率) 域线性插值（与旧版逐行一致）。"""

    def __init__(self, series: list[tuple[float, complex]]) -> None:
        if len(series) < 2:
            raise RuntimeError("校正数据点数不足，无法进行插值")
        cleaned = [(float(f), complex(g)) for f, g in series if f is not None and float(f) > 0.0]
        if len(cleaned) < 2:
            raise RuntimeError("校正数据中有效频点不足，无法进行插值")
        cleaned.sort(key=lambda x: x[0])
        self.freqs = [f for f, _ in cleaned]
        self.xs = [math.log10(f) for f in self.freqs]
        self.reals = [g.real for _, g in cleaned]
        self.imags = [g.imag for _, g in cleaned]

    def evaluate(self, freq: float) -> complex:
        if freq <= 0.0:
            raise RuntimeError("频率必须为正")
        x = math.log10(freq)
        idx = bisect_right(self.xs, x)
        if idx <= 0:
            return self._interp(0, 1, x)
        if idx >= len(self.xs):
            return self._interp(len(self.xs) - 2, len(self.xs) - 1, x)
        return self._interp(idx - 1, idx, x)

    def _interp(self, i0: int, i1: int, x: float) -> complex:
        x0 = self.xs[i0]
        x1 = self.xs[i1]
        if x1 == x0:
            return complex(self.reals[i0], self.imags[i0])
        t = (x - x0) / (x1 - x0)
        real = self.reals[i0] + t * (self.reals[i1] - self.reals[i0])
        imag = self.imags[i0] + t * (self.imags[i1] - self.imags[i0])
        return complex(real, imag)

    def ensure_covers(self, min_freq: float, max_freq: float) -> None:
        src_min = self.freqs[0]
        src_max = self.freqs[-1]
        tol_low = max(1.0, src_min) * 0.05
        tol_high = max(1.0, src_max) * 0.05
        if min_freq < src_min - tol_low or max_freq > src_max + tol_high:
            raise RuntimeError(
                f"测量频率范围 [{min_freq:.6g}, {max_freq:.6g}] 超出校正数据可覆盖范围 [{src_min:.6g}, {src_max:.6g}]"
            )


class FdrExporter:
    def __init__(self, password: str = PASSWORD) -> None:
        self.crypto = FdrCrypto(password)
        self.last_stats: dict[str, float] = {}

    def export_one_file(self, db_path: Path, output_dir: Path, calibration_mode: str = CAL_MODE_NONE) -> int:
        t0 = time.perf_counter()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        db = FdrDatabase(db_path, self.crypto)
        t1 = time.perf_counter()
        try:
            tables = db.tables()
            items = self._collect_export_items(db, tables)
            if not items:
                raise RuntimeError("未找到可导出的子项")

            calibration_ctx: CalibrationContext | None = None
            if calibration_mode != CAL_MODE_NONE:
                if "CalibrationData" not in tables:
                    raise RuntimeError("未找到“校正数据”子项，无法应用所选校准")
                calibration_ctx = self._prepare_calibration_context(db, calibration_mode)

            exported_count = 0
            for item in items:
                csv_path = self._build_output_path(output_dir, db_path.stem, item.display_name)
                apply_ctx = None if item.is_calibration else calibration_ctx
                self._write_table_to_csv(db, item.table_name, csv_path, apply_ctx)
                exported_count += 1
            t2 = time.perf_counter()
            self.last_stats = {
                "decrypt": db.decrypt_seconds,
                "open": db.open_seconds,
                "export": t2 - t1,
                "total": t2 - t0,
            }
            return exported_count
        finally:
            db.close()

    def _collect_export_items(self, db: FdrDatabase, tables: set[str]) -> list[ExportItem]:
        items: list[ExportItem] = []
        if "CalibrationData" in tables:
            items.append(ExportItem("校正数据", "CalibrationData", is_calibration=True))

        if "Sweeps" not in tables:
            return items

        sweep_rows = db.rows("SELECT rowid AS RowId, SweepName FROM Sweeps ORDER BY rowid;")
        for row_id, sweep_name in sweep_rows:
            table_name = f"MeasureData_{int(row_id)}"
            if table_name not in tables:
                continue
            name = str(sweep_name or f"Sweep_{int(row_id)}")
            items.append(ExportItem(name, table_name, is_calibration=False))
        return items

    def _prepare_calibration_context(self, db: FdrDatabase, calibration_mode: str) -> CalibrationContext:
        calibration = self._load_series(db, "CalibrationData")
        if not calibration:
            raise RuntimeError("校正数据为空，无法应用所选校准")

        ctx = CalibrationContext(mode=calibration_mode, calibration=calibration)

        if calibration_mode == CAL_MODE_REF_NORM:
            cleaned = [(float(f), complex(g)) for f, g in calibration if f is not None and float(f) > 0.0]
            cleaned.sort(key=lambda x: x[0])
            ctx.interpolator = ComplexLogInterpolator(calibration)
            if _np is not None:
                freqs = _np.array([f for f, _ in cleaned], dtype=_np.float64)
                ctx.freqs = freqs
                ctx.xs = _np.log10(freqs)
                ctx.reals = _np.array([g.real for _, g in cleaned], dtype=_np.float64)
                ctx.imags = _np.array([g.imag for _, g in cleaned], dtype=_np.float64)
            return ctx

        if calibration_mode == CAL_MODE_DELAY_PHASE:
            ctx.tau = self._estimate_delay_from_short(calibration)
            return ctx

        if calibration_mode == CAL_MODE_SERIES:
            ctx.series_r, ctx.series_l = self._fit_series_rl_from_short(calibration)
            return ctx

        raise RuntimeError(f"未知校准模式：{calibration_mode}")

    @staticmethod
    def _load_series(db: FdrDatabase, table_name: str) -> list[tuple[float, complex]]:
        rows = db.rows(f"SELECT Freq, Real, Image FROM [{table_name}]")
        out: list[tuple[float, complex]] = []
        for freq, real, imag in rows:
            if freq is None or real is None or imag is None:
                continue
            out.append((float(freq), complex(float(real), float(imag))))
        return out

    # ---------------- CSV 写出 ----------------

    def _write_table_to_csv(self, db: FdrDatabase, table_name: str, csv_path: Path,
                            calibration_ctx: CalibrationContext | None) -> None:
        rows = db.rows(f"SELECT Freq, Real, Image FROM [{table_name}]")
        if calibration_ctx is None:
            self._write_plain_rows(csv_path, rows)
            return

        mode = calibration_ctx.mode
        if mode == CAL_MODE_REF_NORM:
            self._write_ref_norm(csv_path, rows, calibration_ctx)
        elif mode == CAL_MODE_DELAY_PHASE:
            self._write_delay_phase(csv_path, rows, calibration_ctx.tau)
        elif mode == CAL_MODE_SERIES:
            self._write_series(csv_path, rows, calibration_ctx.series_r, calibration_ctx.series_l)
        else:
            raise RuntimeError(f"未知校准模式：{mode}")

    @staticmethod
    def _rows_to_arrays(rows: list[tuple]) -> Any:
        arr = _np.array(rows, dtype=_np.float64)
        return arr[:, 0], arr[:, 1] + 1j * arr[:, 2]

    @staticmethod
    def _write_lines(csv_path: Path, lines: list[str]) -> None:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            f.write("Frequency_Hz,S11_Real,S11_Imag\r\n")
            f.write("\r\n".join(lines))
            if lines:
                f.write("\r\n")

    def _write_plain_rows(self, csv_path: Path, rows: list[tuple]) -> None:
        freq_text = self._freq_to_text
        val_text = self._value_to_text
        lines = [
            f"{freq_text(float(freq))},{val_text(float(real))},{val_text(float(imag))}"
            for freq, real, imag in rows
        ]
        self._write_lines(csv_path, lines)

    def _write_ref_norm(self, csv_path: Path, rows: list[tuple], ctx: CalibrationContext) -> None:
        if not rows:
            self._write_lines(csv_path, [])
            return
        if _np is not None and ctx.xs is not None:
            freqs_arr, gamma = self._rows_to_arrays(rows)
            interp = self._vector_log_interp(ctx, freqs_arr)
            mag = _np.abs(interp)
            bad = _np.flatnonzero(mag <= GAMMA_EPS)
            if bad.size:
                f_bad = float(freqs_arr[bad[0]])
                raise RuntimeError(f"频率 {f_bad:.6g} Hz 处校正参考幅值过小，无法做参考归一化")
            min_freq = float(freqs_arr.min())
            max_freq = float(freqs_arr.max())
            src_min = float(ctx.freqs[0])
            src_max = float(ctx.freqs[-1])
            tol_low = max(1.0, src_min) * 0.05
            tol_high = max(1.0, src_max) * 0.05
            if min_freq < src_min - tol_low or max_freq > src_max + tol_high:
                raise RuntimeError(
                    f"测量频率范围 [{min_freq:.6g}, {max_freq:.6g}] 超出校正数据可覆盖范围 [{src_min:.6g}, {src_max:.6g}]"
                )
            corr = -gamma / interp
            self._write_corr_arrays(csv_path, freqs_arr, corr)
            return

        # 纯 Python 回退（与旧版逐行一致）
        assert ctx.interpolator is not None
        ctx.interpolator.ensure_covers(float(min(r[0] for r in rows)), float(max(r[0] for r in rows)))
        val_text = self._value_to_text
        lines = []
        for freq, real, imag in rows:
            f = float(freq)
            gamma_cal = ctx.interpolator.evaluate(f)
            if abs(gamma_cal) <= GAMMA_EPS:
                raise RuntimeError(f"频率 {f:.6g} Hz 处校正参考幅值过小，无法做参考归一化")
            gamma_corr = -complex(float(real), float(imag)) / gamma_cal
            lines.append(f"{self._freq_to_text(f)},{val_text(gamma_corr.real)},{val_text(gamma_corr.imag)}")
        self._write_lines(csv_path, lines)

    @staticmethod
    def _vector_log_interp(ctx: CalibrationContext, freqs_arr: Any) -> Any:
        x = _np.log10(freqs_arr)
        xs = ctx.xs
        n = xs.shape[0]
        idx = _np.searchsorted(xs, x, side="right")
        i1 = _np.clip(idx, 1, n - 1)
        i0 = i1 - 1
        x0 = xs[i0]
        x1 = xs[i1]
        denom = x1 - x0
        safe = _np.where(denom == 0.0, 1.0, denom)
        t = _np.where(denom == 0.0, 0.0, (x - x0) / safe)
        real = ctx.reals[i0] + t * (ctx.reals[i1] - ctx.reals[i0])
        imag = ctx.imags[i0] + t * (ctx.imags[i1] - ctx.imags[i0])
        return real + 1j * imag

    def _write_delay_phase(self, csv_path: Path, rows: list[tuple], tau: float) -> None:
        if not rows:
            self._write_lines(csv_path, [])
            return
        if _np is not None:
            freqs_arr, gamma = self._rows_to_arrays(rows)
            phase_factor = _np.cos(4.0 * _np.pi * freqs_arr * tau) + 1j * _np.sin(4.0 * _np.pi * freqs_arr * tau)
            corr = gamma * phase_factor
            self._write_corr_arrays(csv_path, freqs_arr, corr)
            return
        val_text = self._value_to_text
        lines = []
        for freq, real, imag in rows:
            f = float(freq)
            gamma_meas = complex(float(real), float(imag))
            phase_factor = complex(math.cos(4.0 * math.pi * f * tau), math.sin(4.0 * math.pi * f * tau))
            gamma_corr = gamma_meas * phase_factor
            lines.append(f"{self._freq_to_text(f)},{val_text(gamma_corr.real)},{val_text(gamma_corr.imag)}")
        self._write_lines(csv_path, lines)

    def _write_series(self, csv_path: Path, rows: list[tuple], series_r: float, series_l: float) -> None:
        if not rows:
            self._write_lines(csv_path, [])
            return
        if _np is not None:
            freqs_arr, gamma = self._rows_to_arrays(rows)
            omega = 2.0 * _np.pi * freqs_arr
            z_meas = self._gamma_to_z_vec(gamma)
            z_series = series_r + 1j * (omega * series_l)
            z_corr = z_meas - z_series
            corr = self._z_to_gamma_vec(z_corr)
            self._write_corr_arrays(csv_path, freqs_arr, corr)
            return
        val_text = self._value_to_text
        lines = []
        for freq, real, imag in rows:
            f = float(freq)
            gamma_meas = complex(float(real), float(imag))
            z_meas = self._gamma_to_z(gamma_meas)
            z_series = complex(series_r, 2.0 * math.pi * f * series_l)
            z_corr = z_meas - z_series
            gamma_corr = self._z_to_gamma(z_corr)
            lines.append(f"{self._freq_to_text(f)},{val_text(gamma_corr.real)},{val_text(gamma_corr.imag)}")
        self._write_lines(csv_path, lines)

    def _write_corr_arrays(self, csv_path: Path, freqs_arr: Any, corr: Any) -> None:
        freq_list = freqs_arr.tolist()
        re_list = corr.real.tolist()
        im_list = corr.imag.tolist()
        freq_text = self._freq_to_text
        val_text = self._value_to_text
        lines = [
            f"{freq_text(f)},{val_text(r)},{val_text(i)}"
            for f, r, i in zip(freq_list, re_list, im_list)
        ]
        self._write_lines(csv_path, lines)

    @staticmethod
    def _gamma_to_z_vec(gamma: Any) -> Any:
        denom = 1.0 - gamma
        small = _np.abs(denom) <= GAMMA_EPS
        denom = _np.where(small, complex(GAMMA_EPS, 0.0), denom)
        return Z0 * (1.0 + gamma) / denom

    @staticmethod
    def _z_to_gamma_vec(z_value: Any) -> Any:
        denom = z_value + Z0
        small = _np.abs(denom) <= GAMMA_EPS
        denom = _np.where(small, complex(GAMMA_EPS, 0.0), denom)
        return (z_value - Z0) / denom

    @staticmethod
    def _estimate_delay_from_short(calibration: list[tuple[float, complex]]) -> float:
        pts = [(float(f), -complex(g)) for f, g in calibration if f > 0.0 and abs(g) > GAMMA_EPS]
        if len(pts) < 2:
            raise RuntimeError("校正数据有效点数不足，无法估计时延")

        phases: list[float] = []
        freqs: list[float] = []
        prev: float | None = None
        for f, gamma in pts:
            phase = math.atan2(gamma.imag, gamma.real)
            if prev is not None:
                while phase - prev > math.pi:
                    phase -= 2.0 * math.pi
                while phase - prev < -math.pi:
                    phase += 2.0 * math.pi
            phases.append(phase)
            freqs.append(f)
            prev = phase

        n = len(freqs)
        sx = sum(freqs)
        sy = sum(phases)
        sxx = sum(x * x for x in freqs)
        sxy = sum(x * y for x, y in zip(freqs, phases))
        denom = n * sxx - sx * sx
        if abs(denom) <= 1e-30:
            raise RuntimeError("时延估计失败：频率分布异常")
        slope = (n * sxy - sx * sy) / denom
        tau = -slope / (4.0 * math.pi)
        return tau

    @staticmethod
    def _fit_series_rl_from_short(calibration: list[tuple[float, complex]]) -> tuple[float, float]:
        pts = [(float(f), FdrExporter._gamma_to_z(complex(g))) for f, g in calibration if f > 0.0]
        if len(pts) < 2:
            raise RuntimeError("校正数据有效点数不足，无法拟合串联寄生")

        real_parts = [z.real for _, z in pts]
        series_r = sum(real_parts) / len(real_parts)

        omegas = [2.0 * math.pi * f for f, _ in pts]
        imag_parts = [z.imag for _, z in pts]
        denom = sum(w * w for w in omegas)
        if denom <= 1e-30:
            raise RuntimeError("串联寄生拟合失败：频率分布异常")
        series_l = sum(w * x for w, x in zip(omegas, imag_parts)) / denom
        return series_r, series_l

    @staticmethod
    def _gamma_to_z(gamma: complex) -> complex:
        denom = 1.0 - gamma
        if abs(denom) <= GAMMA_EPS:
            denom = complex(GAMMA_EPS, 0.0)
        return Z0 * (1.0 + gamma) / denom

    @staticmethod
    def _z_to_gamma(z_value: complex) -> complex:
        denom = z_value + Z0
        if abs(denom) <= GAMMA_EPS:
            denom = complex(GAMMA_EPS, 0.0)
        return (z_value - Z0) / denom

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        name = name.strip()
        if not name:
            name = "未命名子项"
        return INVALID_FILENAME_CHARS.sub("_", name)

    def _build_output_path(self, target_dir: Path, file_stem: str, item_name: str) -> Path:
        safe_name = self._sanitize_filename(item_name)
        filename = f"{file_stem}-{safe_name}.csv"
        return self._unique_path(target_dir / filename)

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        index = 2
        while True:
            candidate = parent / f"{stem} ({index}){suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    @staticmethod
    def _freq_to_text(value: float) -> str:
        rounded = round(value)
        if abs(value - rounded) <= max(1e-9, abs(value) * 1e-12):
            return str(int(rounded))
        text = f"{value:.12f}".rstrip("0").rstrip(".")
        return text if text else "0"

    @staticmethod
    def _value_to_text(value: float) -> str:
        return f"{value:.10f}"


# ---------------------------------------------------------------------------
# 图形界面（布局与旧版一致）
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{TITLE} V2极速版（纯Python无依赖）")
        self.geometry("980x580")
        self.minsize(820, 460)

        self.exporter = FdrExporter()
        self.pending_files: list[Path] = []
        self.worker_thread: threading.Thread | None = None
        self.log_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.processing = False

        self.include_subdirs_var = tk.BooleanVar(value=False)
        self.output_to_source_var = tk.BooleanVar(value=False)
        self.calibration_mode_var = tk.StringVar(value=CAL_MODE_NONE)

        self._build_ui()
        self.after(100, self._poll_log_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        row1 = ttk.Frame(root)
        row1.pack(fill=tk.X, pady=(0, 8))
        self.file_button = ttk.Button(row1, text="选择文件", command=self.choose_file)
        self.file_button.pack(side=tk.LEFT)
        self.folder_button = ttk.Button(row1, text="选择文件夹", command=self.choose_folder)
        self.folder_button.pack(side=tk.LEFT, padx=(8, 0))
        self.subdir_check = ttk.Checkbutton(row1, text="包含子目录", variable=self.include_subdirs_var)
        self.subdir_check.pack(side=tk.LEFT, padx=(12, 0))

        row2 = ttk.Frame(root)
        row2.pack(fill=tk.X, pady=(0, 8))
        self.process_button = ttk.Button(row2, text="数据处理", command=self.process_files)
        self.process_button.pack(side=tk.LEFT)
        self.output_check = ttk.Checkbutton(row2, text="输出到源目录", variable=self.output_to_source_var)
        self.output_check.pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(row2, text="校准选项:").pack(side=tk.LEFT, padx=(18, 4))
        self.calibration_combo = ttk.Combobox(
            row2,
            textvariable=self.calibration_mode_var,
            values=CALIBRATION_OPTIONS,
            state="readonly",
            width=16,
        )
        self.calibration_combo.pack(side=tk.LEFT)

        log_frame = ttk.Frame(root)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scroll.set)

    def clear_log(self) -> None:
        self.log_text.delete("1.0", tk.END)

    def log(self, message: str) -> None:
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def choose_file(self) -> None:
        if self.processing:
            return
        path = filedialog.askopenfilename(title="选择工程数据文件", filetypes=FILETYPES)
        if not path:
            return
        self.pending_files = [Path(path)]
        self._show_pending_files()

    def choose_folder(self) -> None:
        if self.processing:
            return
        folder = filedialog.askdirectory(title="选择包含 .fdr/.db8 的文件夹")
        if not folder:
            return
        base = Path(folder)
        self.pending_files = self._collect_files(base, recursive=self.include_subdirs_var.get())
        self._show_pending_files()

    def _collect_files(self, base: Path, recursive: bool) -> list[Path]:
        if recursive:
            candidates = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in {".fdr", ".db8"}]
        else:
            candidates = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in {".fdr", ".db8"}]
        candidates.sort(key=lambda p: str(p).lower())
        return candidates

    def _show_pending_files(self) -> None:
        self.clear_log()
        self.log("待处理.db8/.fdr文件：")
        if not self.pending_files:
            self.log("（未找到文件）")
            return
        for path in self.pending_files:
            self.log(path.name)

    def process_files(self) -> None:
        if self.processing:
            return
        if not self.pending_files:
            messagebox.showwarning("未选择文件", "请先通过“选择文件”或“选择文件夹”指定待处理文件。")
            return

        output_dir: Path | None = None
        if self.output_to_source_var.get():
            self.log("输出位置：各文件源目录")
        else:
            selected = filedialog.askdirectory(title="选择 CSV 输出文件夹")
            if not selected:
                return
            output_dir = Path(selected)
            self.log(f"输出位置：{output_dir}")

        self._set_processing(True)
        worker = threading.Thread(
            target=self._worker_process_files,
            args=(list(self.pending_files), output_dir, self.output_to_source_var.get(), self.calibration_mode_var.get()),
            daemon=True,
        )
        self.worker_thread = worker
        worker.start()

    def _worker_process_files(
        self,
        pending_files: list[Path],
        output_dir: Path | None,
        output_to_source: bool,
        calibration_mode: str,
    ) -> None:
        total = len(pending_files)
        success = 0
        failed = 0

        for db_path in pending_files:
            try:
                target_dir = db_path.parent if output_to_source else output_dir
                if target_dir is None:
                    raise RuntimeError("未确定输出目录")
                count = self.exporter.export_one_file(db_path, target_dir, calibration_mode=calibration_mode)
                success += 1
                stats = self.exporter.last_stats
                self.log_queue.put((
                    "log",
                    f"{db_path.name}：导出 {count} 个 CSV，"
                    f"解密 {stats['decrypt'] + stats['open']:.3f}s，"
                    f"导出 {stats['export']:.3f}s，总计 {stats['total']:.3f}s",
                ))
            except Exception as exc:
                failed += 1
                self.log_queue.put(("log", f"{db_path.name} 转换失败：{exc}"))

        self.log_queue.put(("done", (total, success, failed)))

    def _poll_log_queue(self) -> None:
        while True:
            try:
                kind, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self.log(str(payload))
            elif kind == "done":
                total, success, failed = payload
                self.log(f"转换情况：选择{total}项，成功{success}项，失败{failed}项")
                self._set_processing(False)

        self.after(100, self._poll_log_queue)

    def _set_processing(self, processing: bool) -> None:
        self.processing = processing
        state = tk.DISABLED if processing else tk.NORMAL
        self.file_button.configure(state=state)
        self.folder_button.configure(state=state)
        self.process_button.configure(state=state)
        self.subdir_check.configure(state=state)
        self.output_check.configure(state=state)
        self.calibration_combo.configure(state="disabled" if processing else "readonly")


def _run_cli(paths: list[str]) -> int:
    """命令行批量：输出到各文件源目录，不应用校准。"""
    exporter = FdrExporter()
    lines: list[str] = []
    ok, failed = 0, []
    t_all = time.perf_counter()
    for raw in paths:
        p = Path(raw)
        try:
            count = exporter.export_one_file(p, p.parent, CAL_MODE_NONE)
            stats = exporter.last_stats
            lines.append(f"{p.name}: 导出 {count} 个 CSV，总计 {stats['total']:.3f}s")
            ok += 1
        except Exception as exc:
            failed.append(f"{p.name}: {exc}")
            lines.append(f"{p.name}: 转换失败：{exc}")
    lines.append(f"完成：成功 {ok} 项，失败 {len(failed)} 项，用时 {time.perf_counter() - t_all:.3f}s")

    if sys.stdout is None:  # pythonw 无控制台：弹出汇总对话框
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(f"{TITLE} V2极速版 - 批量结果", "\n".join(lines))
        root.destroy()
    else:
        for line in lines:
            print(line)
    return 0 if not failed else 1


def main() -> None:
    args = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
    if args:
        sys.exit(_run_cli(args))
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
