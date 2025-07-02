import re
from dataclasses import dataclass
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

@dataclass
class ModelInfo:
    generation: int
    model_rank: int
    fitness: float
    path: Path

pattern = re.compile(r"model_rank(\d+)_fitness([0-9]*\.?[0-9]+)\.pth")

def collect_models(base_dir: Path = Path("models")) -> List[ModelInfo]:
    result = []
    for file in base_dir.rglob("model_rank*_fitness*.pth"):
        match = pattern.fullmatch(file.name)
        if not match:
            continue
        rank, fitness = match.groups()

        try:
            generation = int(file.parent.name)
        except ValueError:
            continue

        result.append(ModelInfo(
            generation=generation,
            model_rank=int(rank),
            fitness=float(fitness),
            path=file
        ))
    return result

if __name__ == "__main__":
    models = collect_models()
    if not models:
        print("No models found.")
        exit(1)

    x = [m.generation for m in models]
    y = [m.fitness for m in models]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=25, c='black', marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness per Generation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models_plot.png", dpi=300)
    plt.close()
