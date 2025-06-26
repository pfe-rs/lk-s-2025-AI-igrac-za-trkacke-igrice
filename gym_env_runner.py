from gym_env_custom import CustomEnvGAWithQuads

import pygame

ray_number=7
parametri=6
stanja=4

# === Hyperparameters ===
n_inputs = parametri+stanja+2*ray_number              # Number of rays/sensors
level_file = "clean-codes/levels/8.pkl"


car_params = (5,40,20,([100,200,255]),1500,10,(0,0,0),5)  # Example car: mass, length, width, color, pull, ni
# (self, mass, length, width, color,pull,ni=5,location=(100,100,0.5*math.pi),k=5):
# === Initialize Environment ===w
env = CustomEnvGAWithQuads(n_inputs, level_file, car_params)
env.start_pygame()
state = env.reset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run=False  
                env.close()             
    

    keys = pygame.key.get_pressed()
    action=[False,False,False,False]
    if keys[pygame.K_w]:
        action[0]=True
    if keys[pygame.K_SPACE]:
        action[1]=True
    if keys[pygame.K_a]:
        action[2]=True
    if keys[pygame.K_d]:
        action[3]=True

    # === Step ===
    state, reward, done, _, _ = env.step(action,from_state=True)
    env.render()

    if done:
        print("Episode finished! Resetting environment.")
        state = env.reset()

env.close()
