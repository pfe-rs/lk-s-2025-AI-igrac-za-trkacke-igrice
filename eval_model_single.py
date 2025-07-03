import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from modelArh import *  # Your custom model
from balance_it import load_and_balance_for_double,load_and_balance_for_double_maybe,load_only
from GATrainer import get_last_gen
from Functions import load_all_levels
from GATrainer import env_fn
from gym_env_custom import *
import numpy as np
import time
from Functions import *
import pygame
import time


gb_num=None
lr_num=None







start_time = time.time()

output_file="output8.txt"

car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)

n_inputs=584

every=50


levels_loc="clean-codes/levels/"

# gb_num=500
# lr_num=500

folder_name="models_supervised_maybe_big/"








gb_part="gas_brake_model"
lr_part="steer_model"

ekst=".pkl"

agent_class=CarGameAgentDoubleMaybeSneakyBig
combined_class=CombinedCarGameAgentMaybeReplay
env_class=CustomEnvGAWithQuads

# os.system('clear')


if not(gb_num is None or lr_num is None):
    models_paths = []
    gb_model_loc = folder_name + gb_part + str(gb_num) + ekst
    lr_model_loc = folder_name + lr_part + str(lr_num) + ekst

    # if not os.path.isfile(gb_model_loc) or not os.path.isfile(lr_model_loc):
    #     raise FileNotFoundError(f"Nedostaje model: {gb_model_loc} ili {lr_model_loc}")

    # gb_model = agent_class(n_inputs)
    # lr_model = agent_class(n_inputs)

    # combined_model = combined_class(gb_model, lr_model)
    models_paths.append([gb_model_loc,lr_model_loc])
else:
    models_paths = load_all_models_paths_replay(folder_name,every)

levels_paths = [levels_loc+lvl for lvl in sorted(os.listdir(levels_loc)) if lvl.endswith(".pkl")]

rewards=eval_models_levels(models_paths,levels_paths,env_class,n_inputs,car_params,agent_class,combined_class)

end_time = time.time()
print(f"Execution time: {end_time - start_time:.6f} seconds")

print(rewards)

with open(output_file, "w") as f:
    for idx, reward in enumerate(rewards):
        f.write(f"Model {idx}: {reward:.3f}\n")

