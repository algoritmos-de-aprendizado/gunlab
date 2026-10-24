from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    robocode_home: Path | None = None
    launcher: list[str] | None = None
    quota_bytes: int = 100_000_000
    width: int = 800
    height: int = 600
    rounds: int = 35
    live_rounds: int = 5
    tps: int = 50
    bins: int = 31
    battles_per_opponent: int = 3
    virtual_stride: int = 4
    data_dir: Path = Path("data")
    submissions_dir: Path = Path("submissions")
    results_dir: Path = Path("results")
    tiers: dict[str, list[str]] = field(default_factory=dict)
    arena_opponents: list[str] = field(default_factory=list)

    def require_home(self) -> Path:
        if self.robocode_home is None:
            raise RuntimeError("set [robocode] home in gunlab.toml")
        if not self.robocode_home.is_dir():
            raise RuntimeError(f"Robocode home not found: {self.robocode_home}")
        return self.robocode_home


def load(path: Path) -> Config:
    default = Config()
    if not path.exists():
        return default
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    root = path.resolve().parent
    robocode = raw.get("robocode", {})
    battle = raw.get("battle", {})
    live = raw.get("live", {})
    board = raw.get("board", {})
    data = raw.get("data", {})
    arena = raw.get("arena", {})
    home = robocode.get("home")
    launcher = robocode.get("launcher")
    return Config(
        robocode_home=_resolve(root, home) if home else None,
        launcher=[str(part) for part in launcher] if launcher else None,
        quota_bytes=int(robocode.get("quota_bytes", default.quota_bytes)),
        width=int(battle.get("width", default.width)),
        height=int(battle.get("height", default.height)),
        rounds=int(battle.get("rounds", default.rounds)),
        live_rounds=int(live.get("rounds", default.live_rounds)),
        tps=int(live.get("tps", default.tps)),
        bins=int(board.get("bins", default.bins)),
        battles_per_opponent=int(data.get("battles_per_opponent", default.battles_per_opponent)),
        virtual_stride=int(data.get("virtual_stride", default.virtual_stride)),
        data_dir=_resolve(root, data.get("dir", "data")),
        submissions_dir=_resolve(root, arena.get("submissions", "submissions")),
        results_dir=_resolve(root, arena.get("results", "results")),
        tiers={name: [str(robot) for robot in robots] for name, robots in raw.get("tiers", {}).items()},
        arena_opponents=[str(robot) for robot in arena.get("opponents", [])],
    )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path
