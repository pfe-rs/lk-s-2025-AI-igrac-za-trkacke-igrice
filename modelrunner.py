import sys
from modelArh import CarGameAgent
from gym_env_custom import CustomEnvGAWithQuads
import time
import torch

start_time = time.time()

def env_fn():
    return CustomEnvGAWithQuads(n_inputs, level_file, car_params)

ray_number = 7
parametri = 6
stanja = 4
n_inputs = parametri + stanja + 2 * ray_number
maxsteps=500



if __name__ == "__main__":
    pass
    level_file = "levels/10.pkl"
    car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)

    model_path=sys.argv[1]
    model = CarGameAgent(n_inputs)
    model.load_state_dict(torch.load(model_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    reward=model.run_in_environment(env_fn(), visualize=True, threshold=0.5,maxsteps=10000,device=device)
    # best_model.run_in_environment(env_fn(), visualize=True, threshold=0.5,maxsteps=200)


    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.6f} seconds")
