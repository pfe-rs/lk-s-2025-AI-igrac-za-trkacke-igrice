from gym_env_custom import CustomEnvGAWithQuads
import pygame
from Functions import save_record

# === Parameters ===
ray_number = 7
parametri = 6
stanja = 4

n_inputs = parametri + stanja + 2 * ray_number  # Total input features for the car

level_file = "levels/10.pkl"

# Car params: mass, length, width, color (RGB), pull, ni, location, k
car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)

# Initialize environment
env = CustomEnvGAWithQuads(n_inputs, level_file, car_params)
env.start_pygame()
state = env.reset()

# Recording storage
record = []

number=5

# === Game Loop ===
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

    # Get action based on keys
    action = [False, False, False, False]  # [forward, brake, left, right]
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        action[0] = True
    if keys[pygame.K_SPACE]:
        action[1] = True
    if keys[pygame.K_a]:
        action[2] = True
    if keys[pygame.K_d]:
        action[3] = True

    # Save current state and action 
    record.append([env.state, action])

    # Step environment
    state, reward, done, _, _ = env.step(action)
    env.render()

    # Check end conditions
    if done or not running:
        print("Episode finished! Saving recording...")
        save_record(record, f"clean-codes/recordings/{number}.pkl")
        break

# Cleanup
env.close()
pygame.quit()
print("Recording session ended.")
