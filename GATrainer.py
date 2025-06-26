import sys
import torch
from agent.utils import get_device
from modelArh import CarGameAgent, Population
from gym_env_custom import CustomEnvGAWithQuads
import os

# === Environment Parameters ===
ray_number = 7
parametri = 6
stanja = 4
n_inputs = parametri + stanja + 2 * ray_number

level_file = "levels/11.pkl"
car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)

# Function to create fresh environment per agent run
def env_fn() -> CustomEnvGAWithQuads:
    return CustomEnvGAWithQuads(n_inputs, level_file, car_params)

def get_last_gen(base_dir: str) -> int:
    largest_number = 0

    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)
        return 0

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            current_number = int(item)
            if largest_number is None or current_number > largest_number:
                largest_number = current_number
    return largest_number


# === GA Hyperparameters ===
population_size = 100
num_generations = 10000000
seteps_limit = 5000
mutation_rate = 0.1
mutation_strength = 0.2
elite_fraction = 0.2

if __name__ == "__main__":
    models_dir = sys.argv[1]
    last_gen = get_last_gen(models_dir)
    print("Last generation:", last_gen)

    init_population_path = sys.argv[1]
    population = Population(population_size, n_inputs, mutation_rate, mutation_strength, elite_fraction)

    # If the path has no .pth files, try descending into the latest numbered subdir
    pth_files = [f for f in os.listdir(init_population_path) if f.endswith(".pth")]
    subdirs_exists = False
    if not pth_files:
        subdirs = [d for d in os.listdir(init_population_path) if d.isdigit()]
        if subdirs:
            latest = str(max(map(int, subdirs)))
            init_population_path = os.path.join(init_population_path, latest)
            print(f"Descending into subdir: {init_population_path}")

    if subdirs_exists:
        # Now actually load
        population.models.clear()
        for filename in os.listdir(init_population_path):
            if filename.endswith(".pth"):
                full_path = os.path.join(init_population_path, filename)
                model = CarGameAgent(n_inputs)
                model.load_state_dict(torch.load(full_path))
                population.models.append(model)

    if not population.models:
        raise RuntimeError(f"No models loaded from {init_population_path}")

    # NOTE: Might be useful later
    # models_copy = copy.deepcopy(population.models)
    # for _ in range(4):
    #     population.models.extend(copy.deepcopy(models_copy))
       
    # === GA Loop ===
    generation = 0
    while(1):
        print(f"\n=== Generation {generation} ===")
        
        # Evaluate all models
        population.evaluate_gpu(env_fn, 0.5, seteps_limit, get_device())
        
        # Print best result
        best_model, best_fitness = population.best_model()
        print(f"Best Fitness: {best_fitness:.2f}")

        # Optionally run the best model visually
        population.save_best_models(os.path.join(models_dir, str(last_gen+generation)), int(population_size*elite_fraction))
        # if generation % 10 == 0:
            
        #     print("Visualizing best agent...")
        #     # (self, env, visualize=True, threshold=0.5,maxsteps=500):
        #     best_model.run_in_environment(env_fn(), visualize=True, threshold=0.5,maxsteps=200)

        # Create next generation
        population.next_generation()
        generation += 1
