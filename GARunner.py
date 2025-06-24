from gym_env_custom import CustomEnvGA, CustomEnvGAWithQuads
import random
import pygame
import torch
from modelArh import CarGameAgent

ray_number = 7
parametri = 6
stanja = 4

# === Hyperparameters ===
n_inputs = parametri + stanja + 2 * ray_number  # Number of total inputs
level_file = "levels/10.pkl"

car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)
# mass, length, width, color, pull, ni, location, k

# === Initialize Environment ===
env = CustomEnvGAWithQuads(n_inputs, level_file, car_params)
env.start_pygame()

state = env.reset()

# === Initialize Model ===
model = CarGameAgent(n_inputs)

# Optionally move model to GPU:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension [1, n_inputs]

    action_probs = model(state_tensor)
    threshold = 0.5
    chosen_actions = (action_probs > threshold).int().squeeze(0)  # Remove batch dimension

    # Convert to list of 0s and 1s for environment
    action_list = chosen_actions.tolist()

    # === Step Environment ===
    state, reward, done,steps= env.step(action_list)
    env.render()

    if done:
        print("Episode finished! Resetting environment.")
        state = env.reset()

env.close()
