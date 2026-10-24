from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .model import load_weights

JAVA_DIR = Path(__file__).parent / "java"
# Robocode names robots compiled into robots/ (outside sample.*) as development versions, with a trailing "*".
DEV_SUFFIX = "*"
LOGGER = "gunlab.LoggerBot" + DEV_SUFFIX
ENTRY_PACKAGE = "arena"
STATS = "stats.txt"
TMP = "gunlab_tmp"
QUOTA_KEY = "robocode.robot.filesystem.quota"
_RANK = re.compile(r"^\d+(st|nd|rd|th):\s*")


@dataclass
class Result:
    name: str
    score: float
    bullet_damage: float


class Robocode:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.home = cfg.require_home()
        self.robots = self.home / "robots"
        self.jar = self.home / "libs" / "robocode.jar"
        self.tmp = self.home / TMP
        if not self.jar.is_file():
            raise RuntimeError(f"{self.jar} not found; check [robocode] home")

    def launcher(self) -> list[str]:
        if self.cfg.launcher:
            return list(self.cfg.launcher)
        if os.name == "nt":
            return [str(self.home / "robocode.bat")]
        return ["sh", str(self.home / "robocode.sh")]

    def compile(self, sources: list[Path], classpath: tuple[Path, ...] = ()) -> None:
        if shutil.which("javac") is None:
            raise RuntimeError("javac not found on PATH; install a JDK")
        cp = os.pathsep.join(str(p) for p in (self.jar, *classpath))
        cmd = ["javac", "-encoding", "UTF-8", "-cp", cp, "-d", str(self.robots), *(str(s) for s in sources)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("javac failed:\n" + proc.stderr)
        (self.robots / "robot.database").unlink(missing_ok=True)

    def install_kit(self) -> None:
        self.compile(sorted((JAVA_DIR / "gunlab").glob("*.java")))

    def require_kit(self) -> None:
        for name in ("LoggerBot", "NeuralBot"):
            if not (self.robots / "gunlab" / f"{name}.class").is_file():
                raise RuntimeError("robots not compiled; run 'gunlab setup' first")

    def set_quota(self, quota: int) -> None:
        path = self.home / "config" / "robocode.properties"
        path.parent.mkdir(exist_ok=True)
        lines = path.read_text(encoding="latin-1").splitlines() if path.exists() else []
        lines = [line for line in lines if not line.startswith(QUOTA_KEY + "=")]
        lines.append(f"{QUOTA_KEY}={quota}")
        path.write_text("\n".join(lines) + "\n", encoding="latin-1")

    def robot_available(self, name: str) -> bool:
        cls, _, version = name.partition(" ")
        if version:
            return (self.robots / f"{cls}_{version}.jar").is_file()
        class_file = (self.robots / Path(*cls.split("."))).with_suffix(".class")
        return class_file.is_file() or any(self.robots.glob(f"{cls}_*.jar"))

    def data_dirs(self, robot: str) -> list[Path]:
        *package, cls = robot.split(".")
        relative = Path(*package, f"{cls}.data")
        return [self.robots / relative, self.robots / ".data" / relative]

    def stats_files(self, robot: str) -> list[Path]:
        return [directory / STATS for directory in self.data_dirs(robot.rstrip(DEV_SUFFIX))]

    def clear_stats(self, robot: str) -> None:
        for path in self.stats_files(robot):
            path.unlink(missing_ok=True)

    def read_stats(self, robot: str) -> dict[str, int]:
        """NeuralBot's own counters for the last battle: loaded, shots, hits, waves, would_hit."""
        for path in self.stats_files(robot):
            if path.is_file():
                pairs = (line.split() for line in path.read_text(encoding="ascii").splitlines() if line.strip())
                return {key: int(value) for key, value in pairs}
        return {}

    def forget_us(self) -> None:
        """Delete what opponents saved about our robots, so every battle starts from the same state."""
        own = [self.robots / p for p in ("gunlab", ENTRY_PACKAGE, ".data/gunlab", f".data/{ENTRY_PACKAGE}")]
        for pattern in ("gunlab.*", f"{ENTRY_PACKAGE}.*"):
            for path in self.robots.rglob(pattern):
                if path.is_file() and not any(directory in path.parents for directory in own):
                    path.unlink()

    def install_entries(self, weight_files: list[Path]) -> dict[Path, str]:
        """Compile one NeuralBot subclass per weights file and place its weights; returns file -> robot name."""
        self.require_kit()
        sources = self.tmp / "entries"
        for path in (sources, self.robots / ENTRY_PACKAGE, self.robots / ".data" / ENTRY_PACKAGE):
            shutil.rmtree(path, ignore_errors=True)
        package_dir = sources / ENTRY_PACKAGE
        package_dir.mkdir(parents=True)
        entries: dict[Path, str] = {}
        for weights in weight_files:
            try:
                load_weights(weights)
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"{weights}: {exc}") from None
            cls = entry_class(weights)
            robot = f"{ENTRY_PACKAGE}.{cls}"
            if robot in entries.values():
                raise RuntimeError(f"two submissions map to the robot name {robot}")
            source = f"package {ENTRY_PACKAGE};\n\npublic class {cls} extends gunlab.NeuralBot {{\n}}\n"
            (package_dir / f"{cls}.java").write_text(source, encoding="ascii")
            entries[weights] = robot
        self.compile(sorted(package_dir.glob("*.java")), classpath=(self.robots,))
        for weights, robot in entries.items():
            for directory in self.data_dirs(robot):
                directory.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(weights, directory / "weights.txt")
        return {weights: robot + DEV_SUFFIX for weights, robot in entries.items()}

    def run_battle(self, robots: list[str], rounds: int, live: bool, tps: int | None = None) -> list[Result]:
        self.forget_us()
        self.tmp.mkdir(exist_ok=True)
        battle = self.tmp / "current.battle"
        results = self.tmp / "results.txt"
        log = self.tmp / "last_run.log"
        results.unlink(missing_ok=True)
        battle.write_text(
            "\n".join(
                [
                    f"robocode.battleField.width={self.cfg.width}",
                    f"robocode.battleField.height={self.cfg.height}",
                    f"robocode.battle.numRounds={rounds}",
                    "robocode.battle.gunCoolingRate=0.1",
                    "robocode.battle.rules.inactivityTime=450",
                    "robocode.battle.hideEnemyNames=false",
                    "robocode.battle.selectedRobots=" + ",".join(robots),
                ]
            )
            + "\n",
            encoding="latin-1",
        )
        args = ["-battle", f"{TMP}/current.battle", "-results", f"{TMP}/results.txt"]
        args += ["-tps", str(tps or self.cfg.tps)] if live else ["-nodisplay", "-nosound"]
        proc = subprocess.run(
            [*self.launcher(), *args],
            cwd=self.home,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )
        log.write_text(proc.stdout or "", encoding="utf-8")
        parsed = parse_results(results)
        if proc.returncode != 0 or len(parsed) != len(robots):
            raise RuntimeError(f"battle {' vs '.join(robots)} failed (exit code {proc.returncode}); see {log}")
        return parsed


def entry_class(weights: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", weights.stem)
    if not name or not name[0].isalpha():
        name = "E" + name
    return name[0].upper() + name[1:]


def parse_results(path: Path) -> list[Result]:
    if not path.exists():
        return []
    header: list[str] | None = None
    results: list[Result] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [c.strip() for c in line.split("\t") if c.strip()]
        if not cells:
            continue
        if cells[0].startswith("Robot Name"):
            header = cells
            continue
        if header is None or not _RANK.match(cells[0]):
            continue
        row = dict(zip(header, cells))
        damage = row.get("Bullet Dmg", cells[4] if len(cells) > 4 else "0")
        results.append(Result(_RANK.sub("", cells[0]), _number(cells[1]), _number(damage)))
    return results


def _number(text: str) -> float:
    return float(text.split()[0])
