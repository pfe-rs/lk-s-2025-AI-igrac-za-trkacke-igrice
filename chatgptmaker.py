import math
from ClassesML2 import Line, Level, Car
import pygame
import cv2
import pickle
import os

# === Podešavanja ===
file_root = "clean-codes/levels/"
file_ext = ".pkl"
bias = 0

BACKGROUND_COLOR = [0, 0, 0]
width = 2000
height = 1200
FPS = 60
g = 9.81
snap_distance = 15

# Funkcija za snapovanje pozicije na postojeće krajeve linija
def snap_to_point(pos, points, threshold):
    for point in points:
        if math.hypot(pos[0] - point[0], pos[1] - point[1]) <= threshold:
            return point
    return pos

# Pravljenje direktorijuma ako ne postoji
os.makedirs(file_root, exist_ok=True)

# === Glavni program ===
for i in range(0, 10):
    
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Kreiranje nivoa")

    # === Faza 1: Crtanje zidova ===
    lines = []
    running = True
    is_drawing = False
    start_pos = None

    print("\n--- Faza 1: Crtanje zidova ---")
    print("Levi klik: crtaj linije")
    print("Taster N: preklopno crtanje više linija bez prekida")
    print("Taster Q: kraj crtanja zidova")

    continuous_mode = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    continuous_mode = not continuous_mode
                if event.key == pygame.K_q:
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                endpoints = [(line.x1, line.y1) for line in lines] + [(line.x2, line.y2) for line in lines]
                if not is_drawing:
                    start_pos = snap_to_point(event.pos, endpoints, snap_distance)
                    is_drawing = True
                else:
                    end_pos = snap_to_point(event.pos, endpoints, snap_distance)
                    lines.append(Line(start_pos[0], start_pos[1], end_pos[0], end_pos[1]))
                    if continuous_mode:
                        start_pos = end_pos
                    else:
                        is_drawing = False

        screen.fill(BACKGROUND_COLOR)
        for line in lines:
            line.draw(screen, (255, 255, 255))
        if is_drawing and not continuous_mode and start_pos:
            pygame.draw.line(screen, (200, 200, 200), start_pos, pygame.mouse.get_pos())
        pygame.display.flip()

    # === Faza 2: Postavljanje auta ===
    print("\n--- Faza 2: Postavljanje početne pozicije auta ---")
    print("Klikni dve tačke za pravac auta")

    coords = []
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                coords.append(event.pos)
                print(f"Tačka {len(coords)}: {event.pos}")
                if len(coords) == 2:
                    running = False

        screen.fill(BACKGROUND_COLOR)
        for line in lines:
            line.draw(screen, (255, 255, 255))
        for coord in coords:
            pygame.draw.circle(screen, (255, 0, 0), coord, 5)
        pygame.display.flip()

    x1, y1 = coords[0]
    x2, y2 = coords[1]
    ori = math.atan2(y2 - y1, x2 - x1)

    car = Car(5, 15, 10, [100, 200, 255], 1000, 10, (x1, y1, ori), 5)

    # === Faza 3: Crtanje checkpointova ===
    checkpoints = []
    running = True
    is_drawing = False
    start_pos = None

    print("\n--- Faza 3: Crtanje checkpointova ---")
    print("Levi klik: crtaj checkpointove")
    print("Taster Q: kraj crtanja checkpointova")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                endpoints = [(line.x1, line.y1) for line in checkpoints] + [(line.x2, line.y2) for line in checkpoints]
                if not is_drawing:
                    start_pos = snap_to_point(event.pos, endpoints, snap_distance)
                    is_drawing = True
                else:
                    end_pos = snap_to_point(event.pos, endpoints, snap_distance)
                    checkpoints.append(Line(start_pos[0], start_pos[1], end_pos[0], end_pos[1]))
                    is_drawing = False

        screen.fill(BACKGROUND_COLOR)
        for line in lines:
            line.draw(screen, (255, 255, 255))
        for line in checkpoints:
            line.draw(screen, (0, 255, 0))
        car.show(screen)
        if is_drawing and start_pos:
            pygame.draw.line(screen, (200, 200, 200), start_pos, pygame.mouse.get_pos())
        pygame.display.flip()

    # === Faza 4: Snimanje nivoa i slike ===
    print("\nSnimanje nivoa...")

    screen.fill(BACKGROUND_COLOR)
    for line in lines:
        line.draw(screen, (255, 255, 255))
    for line in checkpoints:
        line.draw(screen, (0, 255, 0))
    car.show(screen)
    pygame.display.flip()

    img_array = pygame.surfarray.array3d(screen)
    img_array = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_array = cv2.flip(img_array, 0)

    image_path = file_root + str(bias + i) + ".png"
    cv2.imwrite(image_path, img_array)
    print(f"Slika sačuvana kao {image_path}")

    level = Level(lines, checkpoints, BACKGROUND_COLOR, (width, height), FPS, g, (x1, y1, ori))

    
    file_path = file_root + str(bias + i) + file_ext
    with open(file_path, "wb") as f:
        pickle.dump(level, f)

    print(f"Nivo sačuvan kao {file_path}")

    pygame.quit()
