import argparse
import pygame
from pathlib import Path
from pygame.font import Font
from pygame.time import Clock
from stable_baselines3 import PPO
from pygame import Color, SurfaceType, Vector2, surfarray

from agent.ppo.env import Env, env_factory

# Constants
maxsteps = 1500
DEFAULT_MODEL_PATH = Path("models/ppo_best/best_model.zip")
DEFAULT_LEVELS_PATH = Path("levels")

# Rendering function
def render(screen: SurfaceType, clock: Clock, font: Font, env: Env):
    pygame.display.flip()
    clock.tick(env.FPS)

    env.level.draw(screen)
    env.car.draw(screen)
    if env.min_opposite_hits:
        hit_a, hit_b = env.min_opposite_hits
        pygame.draw.line(screen, Color(255, 0, 0),
            (hit_a.wall.x1, hit_a.wall.y1),
            (hit_a.wall.x2, hit_a.wall.y2), 3)
        pygame.draw.line(screen, Color(255, 0, 0),
            (hit_b.wall.x1, hit_b.wall.y1),
            (hit_b.wall.x2, hit_b.wall.y2), 3)

    for line in env.midlines.values():
        pygame.draw.line(screen, Color(255, 255, 255), line[0], line[1], 1)
    
    if env.state.direction_cos and env.state.direction_sin:
        start = env.car.pos
        end = start + Vector2(env.state.direction_cos, env.state.direction_sin) * 100
        pygame.draw.line(screen, Color(0, 255, 0), start, end, 2)
    if env.state.car_rotation_cos and env.state.car_rotation_sin:
        start = env.car.pos
        end = start + Vector2(env.state.car_rotation_cos, env.state.car_rotation_sin) * 100
        pygame.draw.line(screen, Color(100, 150, 100), start, end, 2)

    textscore = font.render(f"Score: {env.score}", True, (255, 255, 255))
    textsteps = font.render(f"Steps: {env.steps}", True, (255, 255, 255))

    screen.blit(textscore, (50, 50))
    screen.blit(textsteps, (50, 100))


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
    env.level.FPS = 1500
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