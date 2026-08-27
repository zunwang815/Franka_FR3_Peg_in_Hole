#!/usr/bin/env python3
"""Create report-ready visualizations for the six-hole Task 2 result."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "task2_visualization",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def set_report_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def draw_layout(output_dir: Path) -> Path:
    # Hole 0..2 are the first row, hole 3..5 the second row. Coordinates are
    # centered around the array origin and reported in millimetres.
    centers = [(x, y) for y in (15.0, -15.0) for x in (-30.0, 0.0, 30.0)]
    target = 4
    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    # The teacher randomizes the complete array origin in a 100 mm x 100 mm
    # workspace window, represented here relative to the nominal origin.
    ax.add_patch(
        Rectangle((-50, -50), 100, 100, fill=False, linewidth=2.0,
                  linestyle="--", edgecolor="#4c78a8",
                  label="10 cm × 10 cm array-origin randomization")
    )
    ax.add_patch(
        Rectangle((-45, -30), 90, 60, facecolor="#f2f2f2", edgecolor="none",
                  alpha=0.8, label="2×3 array footprint")
    )
    for index, (x, y) in enumerate(centers):
        color = "#e45756" if index == target else "#59a14f"
        ax.add_patch(Circle((x, y), 11.5, facecolor=color, edgecolor="white",
                            linewidth=1.5, alpha=0.9, zorder=3))
        ax.text(x, y, str(index), color="white", ha="center", va="center",
                weight="bold", zorder=4)

    ax.annotate(
        "30 mm pitch",
        xy=(0, 15), xytext=(15, 24),
        arrowprops={"arrowstyle": "<->", "color": "#333333"},
        ha="center", va="bottom",
    )
    ax.annotate(
        "30 mm pitch",
        xy=(30, -15), xytext=(39, 0),
        arrowprops={"arrowstyle": "<->", "color": "#333333"},
        ha="left", va="center",
    )
    ax.scatter([0], [0], marker="+", s=100, color="#222222", zorder=5)
    ax.text(2, -5, "nominal array origin", color="#222222")
    ax.text(4, -23, "target hole 4", color="#b33d3d", weight="bold")

    ax.set_title("Task 2: six-hole static array and whole-array randomization")
    ax.set_xlabel("x offset from nominal array origin (mm)")
    ax.set_ylabel("y offset from nominal array origin (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-58, 58)
    ax.set_ylim(-58, 58)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    fig.text(
        0.5, 0.015,
        "Each sleeve opening is 23 mm; rows/columns are 30 mm apart. "
        "Target id is supplied as a six-way one-hot observation.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output_dir / "task2_array_layout.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_success_summary(output_dir: Path) -> Path:
    teacher_path = PROJECT_ROOT / "runs/eval/task2_fair_teacher_action0305_gate2_128.json"
    ppo_path = PROJECT_ROOT / "runs/eval/task2_fair_ppo_task1ppo_action02_eval128.json"
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    ppo = json.loads(ppo_path.read_text(encoding="utf-8"))

    # The basic 512/512 result is recorded in Day6_工作记录.md. The two fair
    # protocol values are loaded from the exact JSON files cited by the final
    # report so the figure cannot silently drift away from the reported table.
    labels = [
        "Basic PPO\nno strong disturbance",
        "Fair geometric\nteacher",
        "Fair residual PPO\nmodel 499",
    ]
    totals = [512, int(teacher["episodes"]), int(ppo["episodes"])]
    success = [512, int(teacher["result"]["success"]), int(ppo["result"]["success"])]
    rates = [100.0 * ok / total for ok, total in zip(success, totals)]
    colors = ["#4c78a8", "#f28e2b", "#59a14f"]

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    bars = ax.bar(range(len(labels)), rates, color=colors, width=0.68)
    for bar, ok, total in zip(bars, success, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            101.2,
            f"{ok}/{total}",
            ha="center", va="bottom", weight="bold", fontsize=10,
        )
    ax.set_ylim(0, 108)
    ax.set_ylabel("success rate (%)")
    ax.set_title("Task 2: basic acceptance and final fair disturbance protocol")
    ax.set_xticks(range(len(labels)), labels)
    ax.axhline(90, color="#c44e52", linestyle="--", linewidth=1.5)
    ax.text(
        0.5, 91.2, "project success target ≥90%",
        color="#a43f43", ha="center", va="bottom", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    fig.text(
        0.5, 0.025,
        "Fair protocol: 20 s · fixed XY hole bias σ=0.5 mm · action noise σ=0.0305 · "
        "gain noise σ=5% · teacher gate=2 mm · action delay=0",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = output_dir / "task2_success_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _ratio_from_markdown(path: Path, method: str) -> tuple[int, int]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and method in line:
            match = re.search(r"(\d+)\s*/\s*(\d+)", line)
            if match:
                return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Could not find {method!r} result in {path}")


def draw_final_results_summary(output_dir: Path) -> Path:
    """Plot every acceptance result stated in the final report's top table."""

    task1_path = PROJECT_ROOT / "runs/eval/stress_rl_gain_hole05_bias_500_summary.md"
    task2_teacher_path = PROJECT_ROOT / "runs/eval/task2_fair_teacher_action0305_gate2_128.json"
    task2_ppo_path = PROJECT_ROOT / "runs/eval/task2_fair_ppo_task1ppo_action02_eval128.json"
    task1_teacher = _ratio_from_markdown(task1_path, "几何教师")
    task1_ppo = _ratio_from_markdown(task1_path, "PPO 500")
    task2_teacher_json = json.loads(task2_teacher_path.read_text(encoding="utf-8"))
    task2_ppo_json = json.loads(task2_ppo_path.read_text(encoding="utf-8"))

    panels = [
        (
            "Task 1: single-hole insertion",
            ["Basic\nteacher", "Basic\nresidual PPO", "Stress\nteacher", "Stress\nresidual PPO"],
            [(512, 512), (512, 512), task1_teacher, task1_ppo],
            ["#9ecae1", "#4c78a8", "#f28e2b", "#59a14f"],
        ),
        (
            "Task 2: six-hole array",
            ["Basic\nresidual PPO", "Fair\nteacher", "Fair\nresidual PPO"],
            [
                (512, 512),
                (
                    int(task2_teacher_json["result"]["success"]),
                    int(task2_teacher_json["episodes"]),
                ),
                (
                    int(task2_ppo_json["result"]["success"]),
                    int(task2_ppo_json["episodes"]),
                ),
            ],
            ["#4c78a8", "#f28e2b", "#59a14f"],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharey=True)
    for ax, (title, labels, ratios, colors) in zip(axes, panels):
        rates = [100.0 * success / total for success, total in ratios]
        bars = ax.bar(range(len(labels)), rates, color=colors, width=0.68)
        for bar, (success, total) in zip(bars, ratios):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.1,
                f"{success}/{total}",
                ha="center", va="bottom", fontsize=10, weight="bold",
            )
        ax.axhline(90, color="#c44e52", linestyle="--", linewidth=1.3)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylim(0, 108)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("success rate (%)")
    fig.suptitle("Final report acceptance results: geometric teacher vs residual PPO", fontsize=16)
    fig.text(
        0.5, 0.025,
        "Basic and strong/fair protocols are shown separately; all final protocols use zero action delay.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    path = output_dir / "final_results_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_manifest(output_dir: Path, paths: list[Path]) -> Path:
    all_figures = sorted({*paths, *output_dir.rglob("*.png")})
    manifest = {
        "task": "Task 2: 2x3 six-hole array",
        "no_action_delay": True,
        "figures": [str(path.resolve().relative_to(PROJECT_ROOT)) for path in all_figures],
        "report_protocols": {
            "task2_basic": {
                "success": "512/512",
                "strong_disturbance": False,
            },
            "task2_fair": {
                "teacher_success": "110/128",
                "ppo_success": "128/128",
                "hole_xy_bias_std_mm": 0.5,
                "action_noise_std": 0.0305,
                "action_gain_noise_std_pct": 5.0,
                "teacher_alignment_gate_mm": 2.0,
                "episode_length_s": 20.0,
                "action_delay_steps": 0,
            },
        },
        "sources": [
            "runs/eval/day6_geom/array_stage_random_origin_128.json",
            "runs/ppo/custom/20260818_141826/model_49.pt",
            "runs/eval/task2_fair_teacher_action0305_gate2_128.json",
            "runs/eval/task2_fair_ppo_task1ppo_action02_eval128.json",
            "runs/ppo/custom/20260818_232714/model_499.pt",
            "Day6_工作记录.md",
            "任务1_任务2_总结报告.md",
        ],
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_report_style()
    paths = [
        draw_layout(args.output_dir),
        draw_success_summary(args.output_dir),
        draw_final_results_summary(args.output_dir),
    ]
    manifest = write_manifest(args.output_dir, paths)
    for path in [*paths, manifest]:
        print(path)


if __name__ == "__main__":
    main()
