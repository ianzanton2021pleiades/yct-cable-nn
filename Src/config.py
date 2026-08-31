"""
config.py — 实验配置管理

从 YAML 加载配置，支持 CLI 覆盖。
"""
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str = "unet1d_medium_baseline"
    seed: int = 42
    exp_dir: str = "../AgentsStorage/experiments"


@dataclass
class ModelConfig:
    name: str = "unet1d"
    base_ch: int = 32
    in_channels: int = 2
    out_channels: int = 1
    kernel_size: int = 7
    # V2 CableFormer fields
    transformer_dim: int = 256
    n_transformer_blocks: int = 3
    n_heads: int = 8
    use_cbam: bool = True


@dataclass
class DataConfig:
    manifest: str = "../DataSet/manifest.yaml"
    channels: list = field(default_factory=lambda: ["impulse", "step"])
    augment: bool = True
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1.0e-3
    optimizer: str = "adamw"
    weight_decay: float = 1.0e-4
    scheduler: str = "cosine_warm"
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    amp: bool = True
    early_stop_patience: int = 10


@dataclass
class LossConfig:
    focal_gamma: float = 2.0
    alpha: float = 1.0
    peak_weight: float = 0.3
    # V2 fields
    version: str = "v1"
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    smooth_weight: float = 0.1


@dataclass
class EvalConfig:
    peak_tolerance_m: float = 2.0
    find_peaks_height: float = 0.2
    find_peaks_distance: int = 4
    find_peaks_prominence: float = 0.1


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    device: str = "cuda"
    eval: EvalConfig = field(default_factory=EvalConfig)


def _set_nested(obj: Any, key_path: str, value: str):
    """设置嵌套属性，如 'train.batch_size' → obj.train.batch_size"""
    parts = key_path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)

    final = parts[-1]
    current = getattr(obj, final)

    # 类型转换
    if isinstance(current, bool):
        value = value.lower() in ("true", "1", "yes")
    elif isinstance(current, int):
        value = int(value)
    elif isinstance(current, float):
        value = float(value)
    elif isinstance(current, list):
        value = value.split(",")

    setattr(obj, final, value)


def load_config(yaml_path: str | Path | None = None,
                cli_args: list[str] | None = None) -> Config:
    """
    加载配置: YAML 基础 + CLI 覆盖。

    Args:
        yaml_path: YAML 配置文件路径，None 则使用默认值
        cli_args: CLI 覆盖参数列表，如 ['--train.batch_size', '64']
    """
    cfg = Config()

    # 从 YAML 加载
    if yaml_path is not None:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        for section_name, section_data in raw.items():
            if section_name == "device":
                cfg.device = section_data
                continue
            section = getattr(cfg, section_name, None)
            if section is None or not isinstance(section_data, dict):
                continue
            for k, v in section_data.items():
                if hasattr(section, k):
                    setattr(section, k, v)

    # CLI 覆盖
    if cli_args is None:
        cli_args = []

    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if arg.startswith("--") and "." in arg:
            key = arg[2:]
            if i + 1 < len(cli_args) and not cli_args[i + 1].startswith("--"):
                _set_nested(cfg, key, cli_args[i + 1])
                i += 2
            else:
                _set_nested(cfg, key, "true")
                i += 1
        else:
            i += 1

    return cfg


def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)


if __name__ == "__main__":
    cfg = load_config("Src/configs/default.yaml")
    print(yaml.dump(config_to_dict(cfg), allow_unicode=True, default_flow_style=False))
