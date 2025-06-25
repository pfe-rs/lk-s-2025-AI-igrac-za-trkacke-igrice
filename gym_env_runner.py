from gym_env_custom import CustomEnvGA,CustomEnvGAWithQuads

import random
import pygame

ray_number=7
parametri=6
stanja=4

# === Hyperparameters ===
n_inputs = parametri+stanja+2*ray_number              # Number of rays/sensors
level_file = "levels/10.pkl"
car_params = (5,40,20,([100,200,255]),1500,10,(0,0,0),5)  # Example car: mass, length, width, color, pull, ni
# (self, mass, length, width, color,pull,ni=5,location=(100,100,0.5*math.pi),k=5):
# === Initialize Environment ===w
env = CustomEnvGAWithQuads(n_inputs, level_file, car_params)
env.start_pygame()
state = env.reset()
states_to_save=[]
actions_to_save=[]
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run=False  
                env.close()             
    
    action=[False,False,False,False]
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        action[0]=True
    if keys[pygame.K_SPACE]:
        action[1]=True
    if keys[pygame.K_a]:
        action[2]=True
    if keys[pygame.K_d]:
        action[3]=True
    # keys = [random.choice([True, False]) for _ in range(4)]


    # action=[False,False,False,False]
    # if keys[0]:
    #     action[0]=True
    # if keys[1]:
    #     action[1]=True
    # if keys[2]:
    #     action[2]=True
    # if keys[3]:
    #     action[3]=True
    

    # === Step ===


    state, reward, done, _, _ = env.step(action)
    env.render()

    if done:
        print("Episode finished! Resetting environment.")
        state = env.reset()

env.close()