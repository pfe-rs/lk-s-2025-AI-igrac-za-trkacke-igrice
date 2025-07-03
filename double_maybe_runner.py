import sys
from agent.utils import get_device
from modelArh import *
from gym_env_custom import CustomEnvGAWithQuads
import time
import torch


gb_model_loc="models_supervised_maybe2/gas_brake_model44.pkl"
lr_model_loc="models_supervised_maybe2/steer_model44.pkl"
level_loc="clean-codes/levels/11.pkl"




start_time = time.time()

def env_fn():
    return CustomEnvGAWithQuads(n_inputs, level_file, car_params)

ray_number = 7
parametri = 6
stanja = 4
n_inputs = parametri + stanja + 2 * ray_number
n_inputs=24
maxsteps = 1500



if __name__ == "__main__":
    car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)
    # (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)  
    
    model_path_gb = gb_model_loc
    model_path_lr = lr_model_loc
    level_file =level_loc

    device = get_device()

    model_gas_brake = CarGameAgentDoubleMaybe(n_inputs).to(device)
    model_left_right = CarGameAgentDoubleMaybe(n_inputs).to(device)

    model_gas_brake.load_state_dict(torch.load(model_path_gb))
    model_left_right.load_state_dict(torch.load(model_path_lr))

    model = CombinedCarGameAgentMaybe(model_gas_brake, model_left_right)
    model.eval()

    reward = model.run_in_environment(
        env_fn(), visualize=True, maxsteps=10000, device=device,plotenzi_loc="plotenzi7"
    )

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

    



    
