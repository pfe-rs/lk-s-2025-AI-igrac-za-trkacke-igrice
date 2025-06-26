import sys
import pygame
from pathlib import Path
from stable_baselines3 import PPO
from pygame import surfarray

from agent.ppo.env import env_factory
from Functions import level_loader

# Constants
maxsteps = 1500
default_model_path = Path("./models/ppo_model.zip")
default_levels_path = Path("./levels")

# Rendering function
def render(screen, clock, font, env):
    pygame.display.flip()
    clock.tick(env.FPS)

    env.level.draw(screen, chosen_walls=env.chosen_walls)
    env.level.checkpoints[env.check_number].draw(screen, (100, 0, 100))
    env.car.show(screen)

    textscore = font.render(f"Score: {env.score}", True, (255, 255, 255))
    textsteps = font.render(f"Steps: {env.steps}", True, (255, 255, 255))

    screen.blit(textscore, (50, 50))
    screen.blit(textsteps, (50, 100))

    env.last_screen_array = surfarray.array3d(screen)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 run_model.py <model_path> <level_path>")

    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_model_path
    levels_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_levels_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not levels_path.exists():
        raise FileNotFoundError(f"Level not found at {levels_path}")

    env = env_factory(levels_path)
    level = level_loader(levels_path)

    model = PPO.load(model_path, env=env, device="cpu")

    pygame.init()
    screen = pygame.display.set_mode((level.proportions[0], level.proportions[1]))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    obs, _ = env.reset()

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        # print(f"Step {env.steps} | Reward: {reward:.2f} | Info: {info}")
        render(screen, clock, font, env)

        done = terminated or truncated

    pygame.quit()
