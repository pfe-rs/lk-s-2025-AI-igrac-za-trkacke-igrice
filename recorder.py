from gym_env_custom import CustomEnvGAWithQuads
import pygame
from Functions import save_record
import random

# === Parameters ===
ray_number = 7
parametri = 6
stanja = 4

n_inputs = parametri + stanja + 2 * ray_number  # Total input features for the car 
# level_num=random.randint(0,11)


level_num=10


#record number
number=level_num


level_file = "clean-codes/levels/"+str(level_num)+".pkl"

# Car params: mass, length, width, color (RGB), pull, ni, location, k
car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)

# Initialize environment
env = CustomEnvGAWithQuads(n_inputs, level_file, car_params)
env.start_pygame()
state = env.reset()

# Recording storage
record = []



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
        action[0]=False
    if keys[pygame.K_a]:
        action[2] = True
    if keys[pygame.K_d]:
        action[3] = True

    input_replay=env.get_replay()

    if len(record) <= 20:
        if action[0]:
            record.append([input_replay, action])
    else:
        record.append([input_replay,action])
    

    # Step environment4
    state, reward, done, _, _ = env.step(action,from_state=True)
    env.render()

    # Check end conditions
    if done or not running:
        print("Episode finished! Saving recording...")
        save_record(record, f"clean-codes/recordings_replay/{number}.pkl")
        break

# Cleanup
env.close()
pygame.quit()
print("Recording session ended.")
