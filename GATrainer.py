import sys
import random
import torch
from agent.device import get_device
from modelArh import CarGameAgent, Population
from gym_env_custom import CustomEnvGAWithQuads
import os
import copy

# === Environment Parameters ===
ray_number = 7
parametri = 6
stanja = 4
n_inputs = parametri + stanja + 2 * ray_number

level_file = "levels/10.pkl"
car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)

# Function to create fresh environment per agent run
def env_fn():
    return CustomEnvGAWithQuads(n_inputs, level_file, car_params)

# === GA Hyperparameters ===
population_size = 100
num_generations = 100
mutation_rate = 0.1
mutation_strength = 0.2
elite_fraction = 0.2


if __name__ == "__main__":
    population = Population(population_size, n_inputs, mutation_rate, mutation_strength, elite_fraction)

    init_population_path = sys.argv[1]
    if init_population_path:
        population.models.clear()
        for filename in os.listdir(init_population_path):
            if filename.endswith(".pth"):
                full_path = os.path.join(init_population_path, filename)
                model=CarGameAgent(n_inputs)
                model.load_state_dict(torch.load(full_path))
                population.models.append(model)
        models_copy=copy.deepcopy(population.models)
        for i in range(4):
            population.models.extend(copy.deepcopy(models_copy))

    # === GA Loop ===
    for generation in range(num_generations):
        print(f"\n=== Generation {generation} ===")
        
        # Evaluate all models
        population.evaluate_gpu(env_fn, 0.5, 1500, get_device())
        
        # Print best result
        best_model, best_fitness = population.best_model()
        print(f"Best Fitness: {best_fitness:.2f}")

        # Optionally run the best model visually
        # TODO: Change this hardcoded shit
        population.save_best_models("models/"+str(generation+173),int(population_size*elite_fraction))
        # if generation % 10 == 0:
            
        #     print("Visualizing best agent...")
        #     # (self, env, visualize=True, threshold=0.5,maxsteps=500):
        #     best_model.run_in_environment(env_fn(), visualize=True, threshold=0.5,maxsteps=200)

        # Create next generation
        population.next_generation()