import argparse
import pygame
from pathlib import Path
from pygame.font import Font
from pygame.time import Clock
from stable_baselines3 import PPO
from pygame import Color, SurfaceType, surfarray

from agent.ppo.env import Env, env_factory

# Constants
maxsteps = 1500
DEFAULT_MODEL_PATH = Path("models/ppo_best/best_model.zip")
DEFAULT_LEVELS_PATH = Path("levels")

def draw_guidelines(screen: SurfaceType, env: Env):
    builder = env.guideline
    if not builder or not builder.valid_middlelines:
        return

    arrow_color = Color(0, 255, 0)
    line_color = Color(255, 255, 255)
    arrow_length = 100  # pixels

    for line in builder.valid_middlelines:
        p1, p2 = line
        pygame.draw.line(screen, line_color, p1, p2, 1)

        # Draw direction arrow from midpoint
        mid = (p1 + p2) * 0.5
        dir_vec = (p2 - p1)
        if dir_vec.length() > 0:
            dir_vec = dir_vec.normalize() * arrow_length
            arrow_tip = mid + dir_vec
            pygame.draw.line(screen, arrow_color, mid, arrow_tip, 2)

    # Optional: draw current guidance
    car_pos = env.car.pos
    direction = builder.direction_at(car_pos)
    if direction.length() > 0:
        direction = direction.normalize() * arrow_length
        pygame.draw.line(screen, Color(255, 0, 0), car_pos, car_pos + direction, 3)


# Rendering function
def render(screen: SurfaceType, clock: Clock, font: Font, env: Env):
    pygame.display.flip()
    clock.tick(env.FPS)

    env.level.draw(screen, chosen_walls=env.chosen_walls)
    draw_guidelines(screen, env)
    env.car.draw(screen)
    if env.min_opposite_hits:
        hit_a, hit_b = env.min_opposite_hits
        pygame.draw.line(screen, Color(255, 0, 0),
            (hit_a.wall.x1, hit_a.wall.y1),
            (hit_a.wall.x2, hit_a.wall.y2), 3)
        pygame.draw.line(screen, Color(255, 0, 0),
            (hit_b.wall.x1, hit_b.wall.y1),
            (hit_b.wall.x2, hit_b.wall.y2), 3)

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