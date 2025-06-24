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

level_file = "levels/10.pkl"
car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)

model_path="models/2/model_rank2_fitness200.00.pth"
model = CarGameAgent(n_inputs)
model.load_state_dict(torch.load(model_path))

reward=model.run_in_environment(env_fn(), visualize=True, threshold=0.5,maxsteps=200)
# best_model.run_in_environment(env_fn(), visualize=True, threshold=0.5,maxsteps=200)


end_time = time.time()

print(f"Execution time: {end_time - start_time:.6f} seconds")