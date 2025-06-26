import sys
from agent.device import get_device
from modelArh import CarGameAgentDouble,CombinedCarGameAgent
from gym_env_custom import CustomEnvGAWithQuads
import time
import torch

# python clean-codes/double_runner.py models_supervised/gas_brake_model4.pkl models_supervised/steer_model4.pkl levels/11.pkl

start_time = time.time()

def env_fn():
    return CustomEnvGAWithQuads(n_inputs, level_file, car_params)

ray_number = 7
parametri = 6
stanja = 4
n_inputs = parametri + stanja + 2 * ray_number
maxsteps=1500




if __name__ == "__main__":
    car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)
    if len(sys.argv) < 4:
        print("should provide model and level paths")

    model_path_gb = sys.argv[1]
    model_path_lr = sys.argv[2]
    level_file = sys.argv[3]

    device = get_device()

    model_gas_brake = CarGameAgentDouble(n_inputs).to(device)
    model_left_right = CarGameAgentDouble(n_inputs).to(device)
    
    model_gas_brake.load_state_dict(torch.load(model_path_gb))
    model_left_right.load_state_dict(torch.load(model_path_lr))


    model=CombinedCarGameAgent(model_gas_brake,model_left_right)

    model.eval()
    

    reward=model.run_in_environment(env_fn(), visualize=True, threshold=0.1,maxsteps=10000,device=device)
    # best_model.run_in_environment(env_fn(), visualize=True, threshold=0.5,maxsteps=200)


    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.6f} seconds")
