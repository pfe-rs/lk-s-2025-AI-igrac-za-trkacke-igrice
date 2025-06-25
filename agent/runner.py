import sys
from agent.device import get_device
from modelArh import CarGameAgent
from gym_env_custom import CustomEnvGAWithQuads
import time

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
    if len(sys.argv) < 3:
        print("should provide model and level paths")

    model_path = sys.argv[1]
    level_file = sys.argv[2]