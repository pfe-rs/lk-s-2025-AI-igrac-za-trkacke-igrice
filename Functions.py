import torch
from pathlib import Path
import pickle
from ClassesML2 import *
import os
import tqdm


def level_loader(level_loc: Path | str):
    with open(level_loc, 'rb') as f: 
        level = pickle.load(f)
    return level


def car_from_parameters(parameters):
    # Car(5,40,20,([100,200,255]),1000,10,level.location,3)
    if len(parameters)==9:
        return Car(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7],parameters[8])
    
    else:
        return Car(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7])


def save_record(array, filename):
    """
    Save array clipped by 50 elements at the start and 50 at the end if longer than 100.
    If array has 100 or fewer elements, save the entire array.
    """
    if len(array) > 120:
        to_save = array[:-120]  # Remove 50 from start, 50 from end
    else:
        to_save = array  # Save all if <= 100

    with open(filename, "wb") as f:
        pickle.dump(to_save, f)
    
    print(f"Saved {len(to_save)} elements to {filename}")


def load_record(filename):
    """Load array from pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)



def load_all_levels(levels_folder):
    """
    Loads all .pkl level files from a given folder.

    Args:
        levels_folder (str): Path to the folder containing levels.

    Returns:
        list: List of loaded level objects.
    """
    levels = []

    for filename in os.listdir(levels_folder):
        if filename.endswith(".pkl"):
            path = os.path.join(levels_folder, filename)
            with open(path, "rb") as f:
                level = pickle.load(f)
                levels.append(level)

    # print(f"Loaded {len(levels)} levels from {levels_folder}")
    return levels

def load_all_models_paths_replay(models_folder, every):
    """
    Loads all .pkl model file paths from a given folder, separates gas_brake and steer models,
    and returns a list of [gas_brake_path, steer_path] pairs.

    Args:
        models_folder (str): Path to the folder containing models.
        every (int): Load models at intervals of 'every'.

    Returns:
        list: List of [gas_brake_path, steer_path] pairs.
    """
    paths = []
    
    for filename in sorted(os.listdir(models_folder)):
        if not filename.endswith(".pkl"):
            continue  # Preskoči fajlove koji nisu .pkl
        paths.append(filename)

    # Nađi maksimalni broj modela
    max_number = 0
    for filename in paths:
        num_str = ''.join(filter(str.isdigit, filename))
        if num_str:
            max_number = max(max_number, int(num_str))

    models_paths = [[None, None] for _ in range(max_number + 1)]

    for filename in paths:
        if not filename.endswith(".pkl"):
            continue

        full_path = os.path.join(models_folder, filename)

        if filename.startswith("gas_brake_model"):
            num = int(''.join(filter(str.isdigit, filename)))
            models_paths[num][0] = full_path
        elif filename.startswith("steer_model"):
            num = int(''.join(filter(str.isdigit, filename)))
            models_paths[num][1] = full_path

    # Filtriraj parove koji postoje i uzimaj svaki 'every'-ti
    valid_pairs = [pair for pair in models_paths if pair[0] is not None and pair[1] is not None]
    filtered = [pair for idx, pair in enumerate(valid_pairs) if idx % every == 0]

    print(f"[INFO] Pronađeno {len(filtered)} kombinovanih modela (gas_brake + steer)")

    return filtered

import torch
import numpy as np
from tqdm import tqdm

def eval_models_levels(models_paths, levels_paths, env_class, n_inputs, car_params, agent_class, combined_class):
    """
    Evaluates multiple models across multiple levels and returns mean reward per model.

    Args:
        models_paths (list): List of [gas_brake_path, steer_path] pairs.
        levels_paths (list): Paths to level files.
        env_class (class): Environment class constructor.
        n_inputs (int): Number of inputs for model initialization.
        car_params (dict): Parameters for car/environment setup.
        agent_class (class): Model class for gas_brake and steer models.
        combined_class (class): Class to combine both models.

    Returns:
        list: Mean reward per model (list of floats).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_rewards_per_model = []

    def env_fn(path):
        return env_class(n_inputs, path, car_params)

    print(f"[INFO] Počinjem evaluaciju {len(models_paths)} kombinovanih modela na {len(levels_paths)} staza...")

    for gb_path, lr_path in tqdm(models_paths, desc="Evaluacija modela"):

        gb_model = agent_class(n_inputs)
        lr_model = agent_class(n_inputs)

        gb_model.load_state_dict(torch.load(gb_path, map_location=device))
        lr_model.load_state_dict(torch.load(lr_path, map_location=device))

        gb_model.eval()
        lr_model.eval()

        model = combined_class(gb_model, lr_model)

        model_rewards = []

        for level_path in levels_paths:
            env = env_fn(level_path)
            reward = model.run_in_environment(
                env, visualize=False, maxsteps=1500, device=device
            )
            model_rewards.append(reward / 5)  # Normalizacija po potrebi

        mean_reward = float(np.mean(model_rewards)) if model_rewards else 0.0
        mean_rewards_per_model.append(mean_reward)

    print(f"[INFO] Evaluacija završena.")
    return mean_rewards_per_model
