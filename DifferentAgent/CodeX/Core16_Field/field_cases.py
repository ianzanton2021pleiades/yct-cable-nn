"""保留原 C00–C15 机制；把长度、材料和近端条件放到现场配置中。"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
SOURCE_CASES = HERE.parent / "DG_Core16_Route_Evaluation/core16_cases.yaml"


def build_cases(config: dict, length_m: float) -> list[dict]:
    old = yaml.safe_load(SOURCE_CASES.read_text(encoding="utf-8"))
    old_base = old["cable"]["base_rlgc_per_0p4m"]
    prior = config["body"]
    rng = np.random.default_rng(config["seed"])
    blocks = config["background"]["blocks"]
    edges = np.linspace(0.0, length_m, blocks + 1)
    z_noise, v_noise = rng.normal(size=(2, blocks))
    for i in range(1, blocks):
        z_noise[i] = 0.7 * z_noise[i - 1] + np.sqrt(1 - 0.7**2) * z_noise[i]
        v_noise[i] = 0.7 * v_noise[i - 1] + np.sqrt(1 - 0.7**2) * v_noise[i]

    def position(old_position: float) -> float:
        if old_position == 14.8:
            return length_m * 0.375 - config["local_extent_m"] / 2
        if old_position == 15.2:
            return length_m * 0.375 + config["local_extent_m"] / 2
        return old_position * length_m / 40.0

    cases = []
    for source in old["cases"]:
        changes = [(position(s["start_m"]), position(s["end_m"]), s)
                   for s in source["segments"]]
        boundaries = sorted(set(edges.tolist() + [x for a, b, _ in changes for x in (a, b)]))
        segments = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            midpoint = (a + b) / 2
            block = min(blocks - 1, int(midpoint / length_m * blocks))
            z = prior["z0_ohm"] * np.exp(config["background"]["z_log_sigma"] * z_noise[block])
            v = prior["velocity_m_s"] * np.exp(config["background"]["v_log_sigma"] * v_noise[block])
            raw = next(s for lo, hi, s in changes if lo <= midpoint < hi)
            r, l, g, c = (raw[k] / old_base[k] for k in ("r_ohm", "l_uh", "g_us", "c_pf"))
            local = source["category"] == "local"
            rdc = prior["r_dc_ohm_per_m"]
            rskin = prior["r_skin_ref_ohm_per_m"]
            if local:
                # 局部接触电阻按原案例总欧姆增量保留，不随缺陷展宽重复增加。
                rdc += (raw["r_ohm"] - old_base["r_ohm"]) / config["local_extent_m"]
            else:
                rdc *= r
                rskin *= r
            segment = {
                "start_m": a, "end_m": b,
                "r_dc_ohm_per_m": rdc,
                "r_skin_ref_ohm_per_m": rskin,
                "l_external_h_per_m": z / v * l,
                "c_inf_f_per_m": 1.0 / (z * v) * c,
                "g_dc_s_per_m": prior["g_dc_s_per_m"] * g,
                # G 与 C 分开控制；C单独变化不自动乘大介质损耗导纳。
                "tan_delta": prior["tan_delta"] * g / c,
                "debye_strength": 0.0,
                "debye_corner_hz": prior["debye_corner_hz"],
            }
            segments.append(segment)
        joints = [dict(copy.deepcopy(config["joint"]), position_m=length_m * x)
                  for x in config["joint_positions_fraction"]]
        cases.append({"id": source["id"], "title": source["title"],
                      "category": source["category"], "purpose": source["purpose"],
                      "segments": segments, "joints": joints,
                      "length_m": length_m, "termination": "open"})
    return cases
