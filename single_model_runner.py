import sys
from agent.utils import get_device
from modelArh import CarGameAgentDoubleMaybeFrontMid
from gym_env_custom import CustomEnvGAWithQuads
import time
import torch

# python clean-codes/single_model_runner.py models_supervised/model1.pkl levels/10.pkl

start_time = time.time()

def env_fn():
    return CustomEnvGAWithQuads(n_inputs, level_file, car_params)

ray_number = 7
parametri = 6
stanja = 4
n_inputs = parametri + stanja + 2 * ray_number
n_inputs=584
maxsteps=1500


model_path="models_supervised_absolute/model2001.pkl"
level_file="clean-codes/levels/11.pkl"


if __name__ == "__main__":
    car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)
    if len(sys.argv) < 3:
        print("should provide model and level paths")


    model = CarGameAgentDoubleMaybeFrontMid(n_inputs)
    model.load_state_dict(torch.load(model_path))

    device = get_device()

    reward=model.run_in_environment(env_fn(), visualize=True,maxsteps=10000,device=device,plotenzi_loc="plotenzi2")
    # (self, env, visualize=True, maxsteps=500, device="cuda",startvx=0,startstep=0,plotenzi_loc=None)
    # best_model.run_in_environment(env_fn(), visualize=True, threshold=0.5,maxsteps=200)


    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.6f} seconds")
