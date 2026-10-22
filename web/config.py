"""Đọc configs/web.yaml và cho phép ghi đè bằng --set key.sub=value."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "web.yaml"


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = copy.deepcopy(yaml.safe_load(f) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set cần dạng key.sub=value, nhận được: {item}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)   # '10' -> 10, 'true' -> True, 'null' -> None
    return cfg


def resolve(path: str | os.PathLike) -> Path:
    """Đường dẫn tương đối -> tuyệt đối theo thư mục gốc project."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
