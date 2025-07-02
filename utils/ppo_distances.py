from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Plottable():
    intersections: list[float]
    step: int
    log_file: Path
    log_line: int


def log_to_plottable(log_file: Path, log_line: int, log: dict) -> Plottable:
    step: int = log["step"]
    intersections: list[float] = log["intersections"]
    return Plottable(
        intersections=intersections,
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


if __name__ == "__main__":
    intersections: list[float] = []
    data = collect_logs()
    for d in data:
        intersections.extend(d.intersections)
    
    small_dist = 0.01
    small_dists_count = 0
    for d in data:
        for intersection in d.intersections:
            if intersection > small_dist:
                continue
            small_dists_count += 1
    
    print("min", min(intersections))
    print("avg", sum(intersections) / len(intersections))
    print("max", max(intersections))
    print("total", len(intersections))
    print(f"smaller than {small_dist}", small_dists_count)