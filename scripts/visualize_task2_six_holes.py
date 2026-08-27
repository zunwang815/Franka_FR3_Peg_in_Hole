#!/usr/bin/env python3
"""Capture and replay one verified Task 2 trajectory for each of six holes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUALIZE = PROJECT_ROOT / "scripts" / "visualize.py"
TASK = "Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0"
TASK2_PROTOCOL_CHECKPOINTS = {
    "task2_basic": PROJECT_ROOT / "runs/ppo/custom/20260818_141826/model_49.pt",
    "task2_fair": PROJECT_ROOT / "runs/ppo/custom/20260818_232714/model_499.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", choices=tuple(TASK2_PROTOCOL_CHECKPOINTS), default="task2_fair",
        help="Task-2 report protocol; task2_fair includes 0.5 mm bias, action noise 0.0305 and 5%% gain noise",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Optional checkpoint override; defaults to the selected report protocol model",
    )
    parser.add_argument("--capture_num_envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--replay_fps", type=float, default=24.0)
    parser.add_argument("--initial_hold_seconds", type=float, default=0.4)
    parser.add_argument("--hold_seconds", type=float, default=0.8)
    parser.add_argument(
        "--show_fixture_plate", action=argparse.BooleanOptionalAction, default=True,
        help="Replay-only: keep all six fixed hole sleeves visible",
    )
    parser.add_argument(
        "--screenshot_dir", type=Path,
        default=PROJECT_ROOT / "artifacts" / "task2_visualization" / "six_holes",
    )
    parser.add_argument(
        "--trajectory_output", type=Path, default=None,
        help="Optional persistent combined six-hole .pt trajectory and .json protocol record",
    )
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = TASK2_PROTOCOL_CHECKPOINTS[args.protocol]
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.trajectory_output is not None:
        args.trajectory_output = args.trajectory_output.expanduser().resolve()
    return args


def capture_command(args: argparse.Namespace, hole_id: int, trajectory: Path) -> list[str]:
    return [
        sys.executable, "-u", str(VISUALIZE),
        "--checkpoint", str(args.checkpoint),
        "--protocol", args.protocol,
        "--task", TASK,
        "--geometric_teacher_residual",
        "--episodes", "1",
        "--capture_num_envs", str(args.capture_num_envs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--fixed_target_hole_id", str(hole_id),
        "--_mode", "capture",
        "--_trajectory_file", str(trajectory),
    ]


def replay_command(args: argparse.Namespace, trajectory: Path) -> list[str]:
    command = [
        sys.executable, "-u", str(VISUALIZE),
        "--checkpoint", str(args.checkpoint),
        "--protocol", args.protocol,
        "--task", TASK,
        "--geometric_teacher_residual",
        "--episodes", "6",
        "--device", args.device,
        "--speed", str(args.speed),
        "--replay_fps", str(args.replay_fps),
        "--initial_hold_seconds", str(args.initial_hold_seconds),
        "--hold_seconds", str(args.hold_seconds),
        "--screenshot_dir", str(args.screenshot_dir),
        "--_mode", "replay",
        "--_trajectory_file", str(trajectory),
    ]
    command.append("--show_fixture_plate" if args.show_fixture_plate else "--no-show_fixture_plate")
    return command


def main() -> None:
    args = parse_args()
    args.screenshot_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task2_six_holes_", dir="/tmp") as temp_dir:
        temp = Path(temp_dir)
        episodes = []
        payload_template = None
        for hole_id in range(6):
            trajectory = temp / f"hole_{hole_id}.pt"
            print(f"[TASK2-VIS] Capturing target hole {hole_id} (1/6)", flush=True)
            subprocess.run(capture_command(args, hole_id, trajectory), check=True, cwd=PROJECT_ROOT)
            status_path = trajectory.with_suffix(".json")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("captured") != 1:
                raise RuntimeError(f"Hole {hole_id} did not produce one strict success: {status}")
            payload = torch.load(trajectory, map_location="cpu", weights_only=False)
            payload_template = payload if payload_template is None else payload_template
            episode = payload["episodes"][0]
            episode["result"]["target_hole_id"] = hole_id
            episodes.append(episode)
            print(
                f"[TASK2-VIS] Hole {hole_id} verified: "
                f"depth={episode['result']['depth'] * 1000:.2f}mm",
                flush=True,
            )

        combined = dict(payload_template)
        combined["episodes"] = episodes
        combined["task2_visualization_protocol"] = args.protocol
        combined["task2_visualization_target_sequence"] = list(range(6))
        combined_path = temp / "task2_six_holes_verified.pt"
        torch.save(combined, combined_path)
        combined_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "expected": 6,
                    "captured": 6,
                    "protocol": args.protocol,
                    "target_hole_ids": list(range(6)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if args.trajectory_output is not None:
            args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(combined_path, args.trajectory_output)
            shutil.copy2(
                combined_path.with_suffix(".json"),
                args.trajectory_output.with_suffix(".json"),
            )
            print(
                f"[TASK2-VIS] Persistent trajectory/protocol record: "
                f"{args.trajectory_output}",
                flush=True,
            )
        print("[TASK2-VIS] Replaying hole sequence 0 -> 1 -> 2 -> 3 -> 4 -> 5", flush=True)
        subprocess.run(replay_command(args, combined_path), check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
