"""Discover installed neural-network libraries for the existing FDR GUI.

DG V3 does not ship a trained model.  With no ``*_library.py`` modules this
function returns an empty mapping and the GUI keeps neural analysis disabled.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def discover_libraries() -> dict[str, dict[str, Any]]:
    libraries: dict[str, dict[str, Any]] = {}
    package = __package__ or "ai_libraries"
    for path in sorted(Path(__file__).resolve().parent.glob("*_library.py")):
        module = importlib.import_module(f"{package}.{path.stem}")
        analyze = getattr(module, "analyze", None)
        info = getattr(module, "MODEL_INFO", None)
        if not callable(analyze) or not isinstance(info, dict):
            raise ValueError(f"invalid neural-network library module: {path.name}")
        key = str(info["key"])
        libraries[key] = {"module": module, "info": info, "analyze": analyze}
    return libraries
