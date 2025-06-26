from pathlib import Path
import pickle
import random
from ClassesML2 import Level

class LevelManager:
    loaded_levels: dict[Path, Level] = {}

    def __init__(self, levels_path: Path) -> None:
        if not levels_path.exists():
            raise Exception(f"levels directory path {levels_path} does not exist")
        self.levels_path = levels_path

        for p in levels_path.glob("*.pkl"):
            if p not in self.loaded_levels:
                self.loaded_levels[p] = self._load(p)

    @staticmethod
    def _load(level_path: Path) -> Level:
        with open(level_path, 'rb') as f: 
            level = pickle.load(f)
        return level

    @classmethod
    def random(cls, n: int = 1) -> list[Level]:
        return random.sample(list(cls.loaded_levels.values()), k=n)

    @classmethod
    def get_all(cls) -> list[Level]:
        return list(cls.loaded_levels.values())

    @classmethod
    def get_all_paths(cls) -> list[Path]:
        return list(cls.loaded_levels.keys())

    @classmethod
    def get_by_path(cls, path: Path) -> Level:
        return cls.loaded_levels[path]

