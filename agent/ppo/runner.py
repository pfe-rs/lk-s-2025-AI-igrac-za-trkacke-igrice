import argparse
import pygame
from pathlib import Path
from stable_baselines3 import PPO
from pygame import surfarray

from agent.ppo.env import env_factory

# Constants
maxsteps = 1500
DEFAULT_MODEL_PATH = Path("models/ppo_best/best_model.zip")
DEFAULT_LEVELS_PATH = Path("levels")

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
    parser = argparse.ArgumentParser(description="Run PPO model in environment")
    parser.add_argument(
        "--model_path", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"Path to save/load the PPO model (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--levels_path", type=Path, default=DEFAULT_LEVELS_PATH,
        help=f"Path to the directory containing environment levels (default: {DEFAULT_LEVELS_PATH})"
    )

    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    if not args.levels_path.exists():
        raise FileNotFoundError(f"Level not found at {args.levels_path}")

    env = env_factory(args.levels_path)
    model = PPO.load(args.model_path, env=env, device="cpu")

    pygame.init()
    screen = pygame.display.set_mode((env.level.proportions[0], env.level.proportions[1]))
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

        render(screen, clock, font, env)
        done = terminated or truncated

    pygame.quit()