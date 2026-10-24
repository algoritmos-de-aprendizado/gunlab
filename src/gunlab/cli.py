from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

from . import config as config_module
from .config import Config
from .dataset import TierWriter, clear_logger_files, collect_logger_rows, load_csv
from .robocode import LOGGER, Result, Robocode
from .scoring import ArenaBoard, format_table, offline_board, offline_table, short_name, write_csv


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    cfg = config_module.load(args.config)
    try:
        args.handler(cfg, args)
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from None


def cmd_setup(cfg: Config, args: argparse.Namespace) -> None:
    rc = Robocode(cfg)
    rc.install_kit()
    rc.set_quota(cfg.quota_bytes)
    for folder in (cfg.data_dir, cfg.submissions_dir, cfg.results_dir):
        folder.mkdir(parents=True, exist_ok=True)
    print(f"Robots compiled into {rc.robots / 'gunlab'}")
    opponents = sorted({*cfg.arena_opponents, *(robot for tier in cfg.tiers.values() for robot in tier)})
    missing = [robot for robot in opponents if not rc.robot_available(robot)]
    for robot in missing:
        print(f"warning: {robot} not found in {rc.robots}")
    print(f"{len(opponents) - len(missing)}/{len(opponents)} opponents found")


def cmd_generate(cfg: Config, args: argparse.Namespace) -> None:
    opponents = cfg.tiers.get(args.tier)
    if not opponents:
        raise RuntimeError(f"tier '{args.tier}' is missing or empty in the config")
    rc = Robocode(cfg)
    rc.require_kit()
    battles = args.battles or cfg.battles_per_opponent
    rounds = args.rounds or cfg.rounds
    out = cfg.data_dir / f"{args.tier}.csv"
    clear_logger_files(rc.robots)
    with TierWriter(out, cfg.bins, cfg.virtual_stride) as writer:
        for opponent in opponents:
            for i in range(1, battles + 1):
                rc.run_battle([LOGGER, opponent], rounds, live=args.live, tps=args.tps)
                rows = collect_logger_rows(rc.robots)
                kept = writer.add(rows, opponent)
                print(f"{opponent}  battle {i}/{battles}  rows {kept}/{len(rows)}  total {writer.count}", flush=True)
    print(f"Wrote {writer.count} rows to {out}")


def cmd_score(cfg: Config, args: argparse.Namespace) -> None:
    path = args.data or cfg.data_dir / f"{args.tier}.csv"
    data = load_csv(path)
    if not args.all_waves:
        data = data.subset(data.real == 1)
    if len(data.gf) == 0:
        raise RuntimeError(f"no rows to score in {path}")
    headers, body = offline_table(offline_board(_weights(cfg, args.weights), data, cfg.bins, args.by))
    waves = "waves" if args.all_waves else "real-bullet waves"
    print(f"{path.name}: {len(data.gf)} {waves}, {cfg.bins} GF bins, ranked by {args.by}\n")
    print(format_table(headers, body))
    write_csv(cfg.results_dir / f"offline_{path.stem}.csv", headers, body)


def cmd_arena(cfg: Config, args: argparse.Namespace) -> None:
    opponents = cfg.tiers.get(args.tier, []) if args.tier else cfg.arena_opponents
    if not opponents:
        source = f"tier '{args.tier}'" if args.tier else "[arena] opponents"
        raise RuntimeError(f"{source} is missing or empty in the config")
    rc = Robocode(cfg)
    entries = rc.install_entries(_weights(cfg, args.weights))
    live = not args.headless
    rounds = args.rounds or (cfg.live_rounds if live else cfg.rounds)
    board = ArenaBoard([path.stem for path in entries], opponents)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    total = len(entries) * len(opponents)
    number = 0
    with (cfg.results_dir / f"arena_{stamp}_battles.csv").open("w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        log.writerow(
            ["entry", "opponent", "rounds", "bullet_damage", "score", "shots", "hits", "waves", "would_hit", "loaded", "mode"]
        )
        for path, robot in entries.items():
            for opponent in opponents:
                number += 1
                title = f"Battle {number}/{total}: {path.stem} vs {short_name(opponent)}"
                if live:
                    _show(title, board)
                rc.clear_stats(robot)
                result = _result_for(rc.run_battle([robot, opponent], rounds, live, args.tps), robot)
                stats = rc.read_stats(robot)
                board.record(path.stem, opponent, result.bullet_damage, stats)
                mode = "live" if live else "headless"
                shots, hits, waves, would, loaded = (stats.get(k, 0) for k in ("shots", "hits", "waves", "would_hit", "loaded"))
                log.writerow(
                    [path.stem, opponent, rounds, f"{result.bullet_damage:.2f}", f"{result.score:.2f}",
                     shots, hits, waves, would, loaded, mode]
                )
                f.flush()
                if not live:
                    weights = "" if loaded else "  NO WEIGHTS LOADED"
                    print(
                        f"{title}: hits {hits}/{shots}, would hit {would}/{waves}, "
                        f"{result.bullet_damage:.0f} bullet damage{weights}",
                        flush=True,
                    )
    _show(f"Arena board: hit rate of real bullets, {rounds} rounds per battle", board, clear=live)
    write_csv(cfg.results_dir / f"arena_{stamp}_board.csv", *board.table())


def cmd_watch(cfg: Config, args: argparse.Namespace) -> None:
    rc = Robocode(cfg)
    second = Path(args.opponent)
    duel = second.suffix == ".txt" and second.is_file()
    entries = rc.install_entries([args.weights, second] if duel else [args.weights])
    robots = list(entries.values()) if duel else [entries[args.weights], args.opponent]
    ours = set(entries.values())
    for robot in ours:
        rc.clear_stats(robot)
    results = rc.run_battle(robots, args.rounds or cfg.live_rounds, live=True, tps=args.tps)
    rows = []
    for r in results:
        name = r.name.split(" ")[0]
        stats = rc.read_stats(name) if name in ours else {}
        hits = f"{stats.get('hits', 0)}/{stats.get('shots', 0)}" if stats else "-"
        rows.append([r.name, hits, f"{r.bullet_damage:.0f}", f"{r.score:.0f}"])
    print(format_table(["robot", "hits", "bullet damage", "score"], rows))


def _weights(cfg: Config, given: list[Path]) -> list[Path]:
    files = list(given) or sorted(cfg.submissions_dir.glob("*.txt"))
    if not files:
        raise RuntimeError(f"no weights files given and none found in {cfg.submissions_dir}")
    return files


def _result_for(results: list[Result], robot: str) -> Result:
    for result in results:
        if result.name.split(" ")[0].rstrip("*") == robot.rstrip("*"):
            return result
    raise RuntimeError(f"{robot} is missing from the battle results")


def _show(title: str, board: ArenaBoard, clear: bool = True) -> None:
    if clear:
        os.system("cls" if os.name == "nt" else "clear")
    print(title, end="\n\n")
    print(format_table(*board.table()), end="\n\n", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gunlab", description="Robocode neural-gun dataset generator, offline board and live arena."
    )
    parser.add_argument("--config", type=Path, default=Path("gunlab.toml"), help="course config (default: gunlab.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="compile the robots into Robocode and check the opponents")
    p.set_defaults(handler=cmd_setup)

    p = sub.add_parser("generate", help="run LoggerBot against a tier and write <data>/<tier>.csv")
    p.add_argument("tier")
    p.add_argument("--battles", type=int, help="battles per opponent")
    p.add_argument("--rounds", type=int, help="rounds per battle")
    p.add_argument("--live", action="store_true", help="show the battles")
    p.add_argument("--tps", type=int, help="turns per second with --live")
    p.set_defaults(handler=cmd_generate)

    p = sub.add_parser("score", help="offline board: accuracy and MSE on a CSV")
    p.add_argument("weights", nargs="*", type=Path, help="weights files (default: all in the submissions folder)")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--tier", default="validation", help="score on <data>/<tier>.csv (default: validation)")
    source.add_argument("--data", type=Path, help="score on this CSV")
    p.add_argument("--all-waves", action="store_true", help="also score virtual waves (default: real-bullet waves only)")
    p.add_argument("--by", choices=("hit", "accuracy", "mse"), default="hit", help="ranking metric (default: hit)")
    p.set_defaults(handler=cmd_score)

    p = sub.add_parser("arena", help="arena board: bullet damage against the arena opponents, live by default")
    p.add_argument("weights", nargs="*", type=Path, help="weights files (default: all in the submissions folder)")
    p.add_argument("--headless", action="store_true", help="no display, full speed")
    p.add_argument("--rounds", type=int, help="rounds per battle")
    p.add_argument("--tps", type=int, help="turns per second in live mode")
    p.add_argument("--tier", help="play the robots of this tier instead of [arena] opponents")
    p.set_defaults(handler=cmd_arena)

    p = sub.add_parser("watch", help="one live battle")
    p.add_argument("weights", type=Path)
    p.add_argument("opponent", help="robot name (quote names with a version) or a second weights file")
    p.add_argument("--rounds", type=int, help="rounds per battle")
    p.add_argument("--tps", type=int, help="turns per second")
    p.set_defaults(handler=cmd_watch)
    return parser
