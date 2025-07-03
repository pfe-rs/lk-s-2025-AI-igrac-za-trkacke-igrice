import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import numpy as np
from typing import List

@dataclass
class Plottable():
    angle_diff: float
    reward: float
    step: int
    log_file: Path
    log_line: int

@dataclass
class SinCosPair():
    sin: float
    cos: float

pattern = re.compile(r"")

def get_angle_diff(a: SinCosPair, b: SinCosPair) -> float:
    x1, y1 = a.cos, a.sin
    x2, y2 = b.cos, b.sin
    det = x1 * y2 - y1 * x2
    dot = x1 * x2 + y1 * y2
    return math.atan2(det, dot)

def log_to_plottable(log_file: Path, log_line: int, log: dict) -> Plottable:
    step: int = log["step"]
    reward: float = log["reward"]
    direction_sin: float = log["direction_sin"]
    direction_cos: float = log["direction_cos"]
    car_velocity_x: float = log["car_velocity_x"]
    car_velocity_y: float = log["car_velocity_y"]
    speed = math.hypot(car_velocity_x, car_velocity_y)

    car_velocity_cos: float
    car_velocity_sin: float
    if speed == 0:
        car_velocity_cos = 1.0  # arbitrary, car is not moving
        car_velocity_sin = 0.0
    else:
        car_velocity_cos = car_velocity_x/speed
        car_velocity_sin = car_velocity_y/speed
    
    angle_diff: float = get_angle_diff(
        SinCosPair(sin=direction_sin, cos=direction_cos),
        SinCosPair(sin=car_velocity_sin, cos=car_velocity_cos)
    )
    
    return Plottable(
        angle_diff=angle_diff,
        reward=reward,
        step=step,
        log_file=log_file,
        log_line=log_line
    )

def collect_logs(path: Path = Path("logs")) -> list[Plottable]:
    result: list[Plottable] = []
    for file in path.glob("ppo_state_log_*.json"):
        if not file.is_file():
            continue
        with file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    log = json.loads(line)
                    result.append(log_to_plottable(file, i, log))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # Skip malformed or incomplete lines
    return result

def print_statistics(data: List[Plottable]):
    """Print comprehensive statistical analysis to console"""
    if not data:
        print("No data to analyze")
        return
    
    # Extract arrays for easier computation
    angle_diffs = np.array([p.angle_diff for p in data])
    rewards = np.array([p.reward for p in data])
    steps = np.array([p.step for p in data])
    
    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Basic dataset info
    print(f"Dataset size: {len(data):,} samples")
    print(f"Unique log files: {len(set(p.log_file for p in data))}")
    print()
    
    # Angle difference statistics
    print("ANGLE DIFFERENCE STATISTICS (radians)")
    print("-" * 40)
    print(f"Mean:        {np.mean(angle_diffs):.4f}")
    print(f"Median:      {np.median(angle_diffs):.4f}")
    print(f"Std Dev:     {np.std(angle_diffs):.4f}")
    print(f"Min:         {np.min(angle_diffs):.4f}")
    print(f"Max:         {np.max(angle_diffs):.4f}")
    print(f"Range:       {np.max(angle_diffs) - np.min(angle_diffs):.4f}")
    print(f"25th %ile:   {np.percentile(angle_diffs, 25):.4f}")
    print(f"75th %ile:   {np.percentile(angle_diffs, 75):.4f}")
    print(f"IQR:         {np.percentile(angle_diffs, 75) - np.percentile(angle_diffs, 25):.4f}")
    print()
    
    # Reward statistics
    print("REWARD STATISTICS")
    print("-" * 40)
    print(f"Mean:        {np.mean(rewards):.4f}")
    print(f"Median:      {np.median(rewards):.4f}")
    print(f"Std Dev:     {np.std(rewards):.4f}")
    print(f"Min:         {np.min(rewards):.4f}")
    print(f"Max:         {np.max(rewards):.4f}")
    print(f"Range:       {np.max(rewards) - np.min(rewards):.4f}")
    print(f"25th %ile:   {np.percentile(rewards, 25):.4f}")
    print(f"75th %ile:   {np.percentile(rewards, 75):.4f}")
    print(f"IQR:         {np.percentile(rewards, 75) - np.percentile(rewards, 25):.4f}")
    print()
    
    # Step statistics
    print("STEP STATISTICS")
    print("-" * 40)
    print(f"Mean:        {np.mean(steps):.2f}")
    print(f"Median:      {np.median(steps):.2f}")
    print(f"Std Dev:     {np.std(steps):.2f}")
    print(f"Min:         {np.min(steps)}")
    print(f"Max:         {np.max(steps)}")
    print(f"Range:       {np.max(steps) - np.min(steps)}")
    print()
    
    # Correlation analysis
    correlation = np.corrcoef(angle_diffs, rewards)[0, 1]
    print("CORRELATION ANALYSIS")
    print("-" * 40)
    print(f"Angle diff vs Reward correlation: {correlation:.4f}")
    if abs(correlation) > 0.7:
        strength = "Strong"
    elif abs(correlation) > 0.3:
        strength = "Moderate"
    else:
        strength = "Weak"
    
    direction = "positive" if correlation > 0 else "negative"
    print(f"Correlation strength: {strength} {direction}")
    print()
    
    # Distribution analysis
    print("DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    # Angle difference distribution
    angle_bins = [-math.pi, -math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2, math.pi]
    angle_hist, _ = np.histogram(angle_diffs, bins=angle_bins)
    print("Angle difference distribution:")
    bin_labels = ["[-π, -π/2)", "[-π/2, -π/4)", "[-π/4, 0)", "[0, π/4)", "[π/4, π/2)", "[π/2, π]"]
    for i, (label, count) in enumerate(zip(bin_labels, angle_hist)):
        percentage = (count / len(angle_diffs)) * 100
        print(f"  {label:<12}: {count:>6} ({percentage:>5.1f}%)")
    print()
    
    # Reward distribution
    reward_positive = np.sum(rewards > 0)
    reward_negative = np.sum(rewards < 0)
    reward_zero = np.sum(rewards == 0)
    print("Reward distribution:")
    print(f"  Positive: {reward_positive:>6} ({(reward_positive/len(rewards)*100):>5.1f}%)")
    print(f"  Negative: {reward_negative:>6} ({(reward_negative/len(rewards)*100):>5.1f}%)")
    print(f"  Zero:     {reward_zero:>6} ({(reward_zero/len(rewards)*100):>5.1f}%)")
    print()
    
    # Average reward distribution per angle ranges
    print("AVERAGE REWARD PER ANGLE RANGE")
    print("-" * 40)
    
    # Define angle bins for analysis
    angle_bins = np.array([-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, -math.pi/8, 0, 
                          math.pi/8, math.pi/4, math.pi/2, 3*math.pi/4, math.pi])
    bin_labels = ["[-π, -3π/4)", "[-3π/4, -π/2)", "[-π/2, -π/4)", "[-π/4, -π/8)", 
                  "[-π/8, 0)", "[0, π/8)", "[π/8, π/4)", "[π/4, π/2)", "[π/2, 3π/4)", "[3π/4, π]"]
    
    # Calculate average reward for each angle bin
    for i in range(len(angle_bins) - 1):
        mask = (angle_diffs >= angle_bins[i]) & (angle_diffs < angle_bins[i + 1])
        if i == len(angle_bins) - 2:  # Last bin includes the upper bound
            mask = (angle_diffs >= angle_bins[i]) & (angle_diffs <= angle_bins[i + 1])
        
        bin_rewards = rewards[mask]
        if len(bin_rewards) > 0:
            avg_reward = np.mean(bin_rewards)
            std_reward = np.std(bin_rewards)
            count = len(bin_rewards)
            percentage = (count / len(rewards)) * 100
            print(f"  {bin_labels[i]:<14}: avg={avg_reward:>7.4f}, std={std_reward:>7.4f}, n={count:>5} ({percentage:>4.1f}%)")
        else:
            print(f"  {bin_labels[i]:<14}: No data")
    print()
    
    # Fine-grained analysis around optimal range (±π/8)
    print("FINE-GRAINED ANALYSIS AROUND ZERO (±π/8)")
    print("-" * 40)
    fine_bins = np.linspace(-math.pi/8, math.pi/8, 9)  # 8 bins of π/32 each
    for i in range(len(fine_bins) - 1):
        mask = (angle_diffs >= fine_bins[i]) & (angle_diffs < fine_bins[i + 1])
        if i == len(fine_bins) - 2:  # Last bin includes upper bound
            mask = (angle_diffs >= fine_bins[i]) & (angle_diffs <= fine_bins[i + 1])
        
        bin_rewards = rewards[mask]
        if len(bin_rewards) > 0:
            avg_reward = np.mean(bin_rewards)
            std_reward = np.std(bin_rewards)
            count = len(bin_rewards)
            bin_center = (fine_bins[i] + fine_bins[i + 1]) / 2
            print(f"  [{fine_bins[i]:>6.3f}, {fine_bins[i+1]:>6.3f}): avg={avg_reward:>7.4f}, std={std_reward:>7.4f}, n={count:>4}")
    print()
    
    # Performance insights
    print("PERFORMANCE INSIGHTS")
    print("-" * 40)
    
    # Best/worst angle differences for rewards
    best_reward_idx = np.argmax(rewards)
    worst_reward_idx = np.argmin(rewards)
    
    print(f"Best reward: {rewards[best_reward_idx]:.4f} at angle diff {angle_diffs[best_reward_idx]:.4f}")
    print(f"Worst reward: {rewards[worst_reward_idx]:.4f} at angle diff {angle_diffs[worst_reward_idx]:.4f}")
    
    # Optimal angle difference range
    high_reward_threshold = np.percentile(rewards, 90)
    high_reward_angles = angle_diffs[rewards >= high_reward_threshold]
    if len(high_reward_angles) > 0:
        print(f"Optimal angle diff range (top 10% rewards): [{np.min(high_reward_angles):.4f}, {np.max(high_reward_angles):.4f}]")
        print(f"Mean angle diff for top 10% rewards: {np.mean(high_reward_angles):.4f}")
    
    # Find the angle bin with highest average reward
    best_bin_idx = -1
    best_avg_reward = float('-inf')
    for i in range(len(angle_bins) - 1):
        mask = (angle_diffs >= angle_bins[i]) & (angle_diffs < angle_bins[i + 1])
        if i == len(angle_bins) - 2:
            mask = (angle_diffs >= angle_bins[i]) & (angle_diffs <= angle_bins[i + 1])
        
        bin_rewards = rewards[mask]
        if len(bin_rewards) > 0:
            avg_reward = np.mean(bin_rewards)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_bin_idx = i
    
    if best_bin_idx >= 0:
        print(f"Best performing angle range: {bin_labels[best_bin_idx]} with avg reward {best_avg_reward:.4f}")
    
    print("=" * 60)

if __name__ == "__main__":
    """ Plot
    # x-axis -- direction/velocity angle diff 
    # y-axis -- reward
    """
    data = collect_logs()
    
    # Print statistical analysis to console
    print_statistics(data)
    
    # Create the plot
    x = [p.angle_diff for p in data]
    y = [p.reward for p in data]
    
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, alpha=0.5, s=10)
    plt.xlabel("Direction/Velocity Angle Difference (radians)")
    plt.ylabel("Reward")
    plt.title("Angle Difference vs Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot.png", dpi=150)
    
    print(f"\nPlot saved as 'plot.png'")
    print(f"Total data points plotted: {len(data):,}")