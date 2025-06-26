from gym_env_custom import CustomEnvGAWithQuads
import pickle
import pygame

# === Parameters ===
ray_number = 7
parametri = 6
stanja = 4

n_inputs = parametri + stanja + 2 * ray_number
level_file = "clean-codes/levels/8.pkl"
car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)

# === Load Recording ===
record_file = "clean-codes/recordings2/4.pkl"  # Change to your recording file

with open(record_file, "rb") as f:
    record = pickle.load(f)

print(f"Loaded {len(record)} recorded steps from {record_file}")

# === Initialize Environment ===
env = CustomEnvGAWithQuads(n_inputs, level_file, car_params)
env.start_pygame()
state = env.reset()

# === Replay Loop ===
step_index = 0
running = True
clock = pygame.time.Clock()

while running and step_index < len(record):
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            running = False
            break

    # Get recorded action
    _, action = record[step_index]

    # Step environment with recorded action
    state, reward, done, _, _ = env.step(action)
    env.render()
    step_index += 1

    # Optional: Limit FPS to slow down playback
    clock.tick(30)

    if done:
        print("Episode finished! Resetting environment.")
        state = env.reset()
        step_index = 0  # Start replay from beginning if desired

env.close()
pygame.quit()
print("Replay ended.")
