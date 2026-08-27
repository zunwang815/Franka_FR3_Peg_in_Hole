#!/usr/bin/env python3
"""Compatibility launcher for report-protocol trajectory capture and GUI replay.

The former implementation selected a removed direct-RL checkpoint, used the
legacy IK-Abs task, and did not actually encode an MP4. This launcher now
delegates to ``visualize.py`` so every captured motion uses one of the final
report protocols and records its disturbance metadata.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZE = PROJECT_ROOT / "scripts" / "visualize.py"
PROTOCOLS = ("task1_basic", "task1_stress", "task2_basic", "task2_fair")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture and replay a final report protocol with screenshots and trajectory data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--protocol", choices=PROTOCOLS, default="task2_fair")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--capture_num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory; defaults to artifacts/protocol_replay/<protocol>",
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.capture_num_envs <= 0:
        parser.error("--episodes and --capture_num_envs must be positive")
    if args.output is None:
        args.output = PROJECT_ROOT / "artifacts" / "protocol_replay" / args.protocol
    args.output = args.output.expanduser().resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-u",
        str(VISUALIZE),
        "--protocol", args.protocol,
        "--episodes", str(args.episodes),
        "--capture_num_envs", str(args.capture_num_envs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--speed", str(args.speed),
        "--screenshot_dir", str(args.output / "screenshots"),
        "--trajectory_output", str(args.output / f"{args.protocol}.pt"),
    ]
    if args.checkpoint is not None:
        command.extend(("--checkpoint", str(args.checkpoint)))
    print(
        "[RECORD] Capturing the disturbed physical trajectory first; the GUI "
        "then replays those verified states. No second disturbance is applied.",
        flush=True,
    )
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
