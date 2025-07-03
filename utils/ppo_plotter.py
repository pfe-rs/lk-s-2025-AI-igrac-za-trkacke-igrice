import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass
class SinCosPair:
    sin: float
    cos: float


def get_angle_diff(a: SinCosPair, b: SinCosPair) -> float:
    x1, y1 = a.cos, a.sin
    x2, y2 = b.cos, b.sin
    det = x1 * y2 - y1 * x2
    dot = x1 * x2 + y1 * y2
    return math.atan2(det, dot)


def parse_logs(log_dir: Path = Path("logs")):
    steps = []
    rewards = []
    speeds = []
    angle_diffs = []
    min_wall_dists = []

    for file in log_dir.glob("ppo_state_log_*.json"):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log = json.loads(line)

                    step = log["step"]
                    reward = log["reward"]
                    vx, vy = log["car_velocity_x"], log["car_velocity_y"]
                    speed = math.hypot(vx, vy)

                    if speed == 0:
                        cos_v, sin_v = 1.0, 0.0
                    else:
                        cos_v, sin_v = vx / speed, vy / speed

                    angle_diff = get_angle_diff(
                        SinCosPair(log["direction_sin"], log["direction_cos"]),
                        SinCosPair(sin_v, cos_v)
                    )

                    min_wall = min(log["intersections"])

                    steps.append(step)
                    rewards.append(reward)
                    speeds.append(speed)
                    angle_diffs.append(angle_diff)
                    min_wall_dists.append(min_wall)

                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

    return steps, rewards, speeds, angle_diffs, min_wall_dists

def plot_all(steps, rewards, speeds, angle_diffs, wall_dists, output_path="plot.png"):
    import numpy as np

    def preprocess(steps, values):
        # Sort by step
        combined = sorted(zip(steps, values), key=lambda x: x[0])
        s_sorted, v_sorted = zip(*combined)

        # Insert NaNs on step reset to break lines
        s_final, v_final = [], []
        prev = s_sorted[0]
        for s, v in zip(s_sorted, v_sorted):
            if s < prev:
                s_final.append(np.nan)
                v_final.append(np.nan)
            s_final.append(s)
            v_final.append(v)
            prev = s
        return s_final, v_final

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    labels = ["Reward", "Speed (m/s)", "Angle Diff (rad)", "Wall Dist (m)"]
    values = [rewards, speeds, angle_diffs, wall_dists]
    colors = ["tab:green", "tab:blue", "tab:orange", "tab:red"]

    for ax, label, val, color in zip(axs, labels, values, colors):
        s_proc, v_proc = preprocess(steps, val)
        ax.plot(s_proc, v_proc, color=color, linewidth=0.8, alpha=0.7)
        ax.set_ylabel(label)
        ax.legend([label])

    axs[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")



if __name__ == "__main__":
    steps, rewards, speeds, angle_diffs, wall_dists = parse_logs()
    plot_all(steps, rewards, speeds, angle_diffs, wall_dists)
