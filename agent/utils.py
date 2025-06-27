from pathlib import Path
import pickle
import random

import torch
from ClassesML2 import Level


class LevelManager:
    loaded_levels: dict[Path, 'Level'] = {}
    
    def __init__(self, levels_path: Path) -> None:
        if not levels_path.exists():
            raise Exception(f"levels path {levels_path} does not exist")
        self.levels_path = levels_path
        
        # Clear and repopulate the class variable
        LevelManager.loaded_levels = {}
        
        if levels_path.is_file() and levels_path.suffix == ".pkl":
            LevelManager.loaded_levels[levels_path] = self._load(levels_path)
        elif levels_path.is_dir():
            print("dir")
            for p in levels_path.glob("*.pkl"):
                print(f"found {p}")
                if p not in LevelManager.loaded_levels:
                    LevelManager.loaded_levels[p] = self._load(p)
        else:
            raise Exception(f"Provided path {levels_path} is neither a .pkl file nor a directory")
    
    @staticmethod
    def _load(level_path: Path) -> 'Level':
        with open(level_path, 'rb') as f: 
            level = pickle.load(f)
        return level
    
    @classmethod
    def random(cls, n: int = 1) -> list['Level']:
        levels = list(cls.loaded_levels.values())
        if n > len(levels):
            raise ValueError(f"Requested {n} levels, but only {len(levels)} loaded.")
        if len(cls.loaded_levels) == 1:
            return list(cls.loaded_levels.values())
        return random.sample(levels, k=n)
    
    @classmethod
    def get_all(cls) -> list['Level']:
        return list(cls.loaded_levels.values())
    
    @classmethod
    def get_all_paths(cls) -> list[Path]:
        return list(cls.loaded_levels.keys())
    
    @classmethod
    def get_by_path(cls, path: Path) -> 'Level':
        return cls.loaded_levels[path]

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
