from gym_env_custom import CustomEnvGA,CustomEnvGAWithQuads
import random
import time
import pygame

ray_number = 7
parametri = 6
stanja = 4

use_pygame=False 

start_time = time.time()

# === Hyperparameters ===
n_inputs = parametri + stanja + 2 * ray_number
level_file = "levels/10.pkl"
car_params = (5, 40, 20, [100, 200, 255], 1500, 10, (0, 0, 0), 5)

env = CustomEnvGA(n_inputs, level_file, car_params)
if use_pygame:
    env.start_pygame()

state = env.reset()

maxsteps = 10000
running = True

use_pygame=False


while running:
    if use_pygame:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # handle window close
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        keys = pygame.key.get_pressed()
        action = [False, False, False, False]
        if keys[pygame.K_w]:
            action[0] = True
        if keys[pygame.K_SPACE]:
            action[1] = True
        if keys[pygame.K_a]:
            action[2] = True
        if keys[pygame.K_d]:
            action[3] = True
        
        env.render()
    else:
        action=[False,False,False,False]

    # === Step ===
    state, reward, done, steps = env.step(action)
    


    if done:
        print("Episode finished! Resetting environment.")
        state = env.reset()

    if steps >= maxsteps:
        running = False

env.close()
end_time = time.time()

print(f"Execution time: {end_time - start_time:.6f} seconds")
