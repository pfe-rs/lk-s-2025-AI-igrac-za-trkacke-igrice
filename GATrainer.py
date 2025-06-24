import random
import torch
from modelArh import Population
from gym_env_custom import CustomEnvGAWithQuads

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
population_size = 10
num_generations = 3
mutation_rate = 0.1
mutation_strength = 0.2
elite_fraction = 0.2

# === Initialize Population ===
population = Population(population_size, n_inputs, mutation_rate, mutation_strength, elite_fraction)

device = "cuda" if torch.cuda.is_available() else "cpu"

# === GA Loop ===
for generation in range(num_generations):
    print(f"\n=== Generation {generation} ===")
    
    # Evaluate all models
    population.evaluate(env_fn,False,0.5,200,device)
    
    # Print best result
    best_model, best_fitness = population.best_model()
    print(f"Best Fitness: {best_fitness:.2f}")

    # Optionally run the best model visually
    population.save_best_models("models/"+str(generation),int(population_size*elite_fraction))
    # if generation % 10 == 0:
        
    #     print("Visualizing best agent...")
    #     # (self, env, visualize=True, threshold=0.5,maxsteps=500):
    #     best_model.run_in_environment(env_fn(), visualize=True, threshold=0.5,maxsteps=200)

    # Create next generation
    population.next_generation()
