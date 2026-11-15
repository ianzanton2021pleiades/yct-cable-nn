"""Reader for the field instrument's .fdr files (RC4-obfuscated SQLite).

Format established by the 2026-10 survey: every 4096-byte page is XORed with a
fresh RC4 keystream of the key SHA1("CZY")[:16]; after decryption the file is a
standard SQLite database containing per-phase S11 sweeps, the embedded clamp
short-calibration, and the vendor's own precomputed impulse/step responses.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile

import numpy as np


def _rc4_crypt(key: bytes, data: bytes) -> bytes:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) & 0xFF
        s[i], s[j] = s[j], s[i]
    out = bytearray(len(data))
    i = j = 0
    for n, byte in enumerate(data):
        i = (i + 1) & 0xFF
        j = (j + s[i]) & 0xFF
        s[i], s[j] = s[j], s[i]
        out[n] = byte ^ s[(s[i] + s[j]) & 0xFF]
    return bytes(out)


def decrypt_fdr(path: str) -> bytes:
    raw = open(path, "rb").read()
    key = hashlib.sha1(b"CZY").digest()[:16]
    out = bytearray()
    for off in range(0, len(raw), 4096):
        out += _rc4_crypt(key, raw[off:off + 4096])
    return bytes(out)


def _read_table(cur, name: str):
    cur.execute(f"SELECT * FROM {name}")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return cols, rows


def read_fdr(path: str) -> dict:
    """Return dict: sweeps (per phase), calibration, responses, settings."""
    plain = decrypt_fdr(path)
    fd, tmp = tempfile.mkstemp(suffix=".db")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(plain)
        con = sqlite3.connect(tmp)
        cur = con.cursor()
        out: dict = {"path": path}
        try:
            _, rows = _read_table(cur, "MeasureSetting")
            out["setting"] = rows[0] if rows else None
            _, rows = _read_table(cur, "Sweeps")
            out["sweeps"] = rows
            _, rows = _read_table(cur, "Nameplate")
            out["nameplate"] = rows[0] if rows else None
        except sqlite3.OperationalError:
            pass
        # sweeps row: (SweepName, ?, ?, IsMeasured, Fstart, Fstop, Points,
        #              Wave_Velocity, X_Min, X_Max, ?, X_Offset, ?, IsCableMeasured, Remark)
        measured = [r for r in out.get("sweeps", [])
                    if str(r[3]).strip().lower() == "true" or r[3] == 1]
        out["phases"] = []
        for idx, r in enumerate(measured, start=1):
            name = r[0]
            phase = {"name": name, "fstart": r[4], "fstop": r[5], "points": r[6],
                     "wave_velocity": r[7], "x_max_m": r[9]}
            for table, key in ((f"MeasureData_{idx}", "s11"),
                               (f"ImpulseResponse_{idx}", "impulse"),
                               (f"StepResponse_{idx}", "step")):
                try:
                    cols, rows = _read_table(cur, table)
                    phase[key] = (cols, rows)
                except sqlite3.OperationalError:
                    pass
            out["phases"].append(phase)
        try:
            _, rows = _read_table(cur, "CalibrationData")
            out["calibration"] = rows
        except sqlite3.OperationalError:
            out["calibration"] = None
        con.close()
    finally:
        os.remove(tmp)
    return out


def phase_arrays(phase: dict) -> dict:
    """Convert raw rows to numpy arrays: freq, s11 (complex), impulse, step."""
    result = {}
    cols, rows = phase["s11"]
    ic = {c.lower(): i for i, c in enumerate(cols)}
    arr = np.asarray(rows, dtype=np.float64)
    freq = arr[:, ic["freq"]]
    order = np.argsort(freq)
    freq = freq[order]
    s11 = arr[order, ic["real"]] + 1j * arr[order, ic.get("image", ic.get("imag", 2))]
    result["freq"] = freq
    result["s11"] = s11
    for key in ("impulse", "step"):
        if key in phase:
            cols, rows = phase[key]
            ic = {c.lower(): i for i, c in enumerate(cols)}
            arr = np.asarray(rows, dtype=np.float64)
            result[key + "_dist"] = arr[:, ic["distance"]]
            result[key + "_rho"] = arr[:, ic["rho"]]
    return result
