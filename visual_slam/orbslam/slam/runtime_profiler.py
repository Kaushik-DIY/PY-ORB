"""
Lightweight runtime profiling helpers for the RGB-D ORB-SLAM pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import csv
import json
import time
from pathlib import Path


class RuntimeProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._active: dict[str, float] = {}
        self._stats: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {
                "calls": 0,
                "total_sec": 0.0,
                "mean_sec": 0.0,
                "max_sec": 0.0,
                "last_sec": 0.0,
            }
        )

    def start(self, section_name: str) -> None:
        if not self.enabled:
            return
        self._active[section_name] = time.perf_counter()

    def stop(self, section_name: str) -> float:
        if not self.enabled:
            return 0.0
        start = self._active.pop(section_name, None)
        if start is None:
            return 0.0
        elapsed = max(0.0, time.perf_counter() - start)
        self.record(section_name, elapsed)
        return elapsed

    def record(self, section_name: str, elapsed_sec: float) -> float:
        if not self.enabled:
            return 0.0
        elapsed = max(0.0, float(elapsed_sec))
        stats = self._stats[section_name]
        calls = int(stats["calls"]) + 1
        total = float(stats["total_sec"]) + elapsed
        stats["calls"] = calls
        stats["total_sec"] = total
        stats["mean_sec"] = total / max(calls, 1)
        stats["max_sec"] = max(float(stats["max_sec"]), elapsed)
        stats["last_sec"] = elapsed
        return elapsed

    @contextmanager
    def section(self, section_name: str):
        if not self.enabled:
            yield self
            return
        start = time.perf_counter()
        try:
            yield self
        finally:
            self.record(section_name, time.perf_counter() - start)

    def to_dict(self) -> dict[str, dict[str, float | int | str]]:
        rows = {}
        for section_name in sorted(self._stats.keys()):
            stats = self._stats[section_name]
            rows[section_name] = {
                "section": section_name,
                "calls": int(stats["calls"]),
                "total_sec": float(stats["total_sec"]),
                "mean_sec": float(stats["mean_sec"]),
                "max_sec": float(stats["max_sec"]),
                "last_sec": float(stats["last_sec"]),
            }
        return rows

    def rows(self) -> list[dict[str, float | int | str]]:
        return list(self.to_dict().values())

    def write_csv(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["section", "calls", "total_sec", "mean_sec", "max_sec", "last_sec"]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows():
                writer.writerow(row)
        return path

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "enabled": bool(self.enabled),
            "sections": self.rows(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path
