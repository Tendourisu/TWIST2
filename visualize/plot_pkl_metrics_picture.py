#!/usr/bin/env python3
"""Plot selected motion metrics from a pkl file."""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from pose.utils.motion_lib_pkl import MotionLib
from pose.utils.torch_utils import euler_from_quaternion


def _extract_motion_tensors(motion_lib: MotionLib, motion_idx: int = 0) -> Dict[str, torch.Tensor]:
    """Return tensors for a single motion clip by index."""
    start_idx = int(motion_lib._motion_start_idx[motion_idx].item())
    num_frames = int(motion_lib._motion_num_frames[motion_idx].item())
    end_idx = start_idx + num_frames

    root_pos = motion_lib._motion_root_pos[start_idx:end_idx]
    root_vel = motion_lib._motion_root_vel[start_idx:end_idx]
    root_ang_vel = motion_lib._motion_root_ang_vel[start_idx:end_idx]
    root_rot = motion_lib._motion_root_rot[start_idx:end_idx]
    dt = float(motion_lib._motion_dt[motion_idx].item())

    return {
        "root_pos": root_pos.cpu(),
        "root_vel": root_vel.cpu(),
        "root_ang_vel": root_ang_vel.cpu(),
        "root_rot": root_rot.cpu(),
        "dt": torch.tensor(dt),
    }


def _build_metric_series(motion_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Assemble the six requested metric series."""
    root_pos = motion_tensors["root_pos"]
    root_vel = motion_tensors["root_vel"]
    root_ang_vel = motion_tensors["root_ang_vel"]
    root_rot = motion_tensors["root_rot"]

    euler_angles = euler_from_quaternion(root_rot)

    return {
        "root_x_vel": root_vel[:, 0],
        "root_y_vel": root_vel[:, 1],
        "root_z": root_pos[:, 2],
        "root_roll": euler_angles[:, 0],
        "root_pitch": euler_angles[:, 1],
        "root_yaw_vel": root_ang_vel[:, 2],
    }


def _plot_metrics(time_axis: torch.Tensor, metrics: Dict[str, torch.Tensor], output_path: Path) -> None:
    """Plot metrics over time and save to file."""
    metric_names: List[str] = list(metrics.keys())
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 2.5 * num_metrics), sharex=True)

    if num_metrics == 1:
        axes = [axes]

    for ax, name in zip(axes, metric_names):
        values = metrics[name].numpy()
        ax.plot(time_axis.numpy(), values, label=name)
        ax.set_ylabel(name)
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot root motion metrics from a single pkl file.")
    parser.add_argument("pkl_path", type=Path, help="路径：包含单条 motion 的 pkl 文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pkl_path = args.pkl_path.resolve()

    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} 不存在")

    device = torch.device("cpu")
    motion_lib = MotionLib(
        motion_file=str(pkl_path),
        device=device,
        motion_decompose=False,
        motion_smooth=False,
    )

    if motion_lib.num_motions() != 1:
        print(f"警告：检测到 {motion_lib.num_motions()} 条 motion，将默认使用第一条。")

    motion_tensors = _extract_motion_tensors(motion_lib, motion_idx=0)
    metrics = _build_metric_series(motion_tensors)

    dt = float(motion_tensors["dt"].item())
    time_axis = torch.arange(metrics["root_x_vel"].shape[0], dtype=torch.float32) * dt

    output_path = pkl_path.parent / f"{pkl_path.stem}_metrics.png"
    _plot_metrics(time_axis, metrics, output_path)
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()

## python visualize/plot_pkl_metrics_picture.py /home/zzhang/zhd/TWIST2/assets/example_motions/0807_yanjie_walk_010.pkl