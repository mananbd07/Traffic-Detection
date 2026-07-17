import pygame
import numpy as np
from stable_baselines3 import PPO

from env.traffic_env import TrafficEnv

# =========================
# 🎮 INIT
# =========================
pygame.init()

WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Traffic Quantum Pro")

font = pygame.font.SysFont(None, 24)

# =========================
# 🚦 ENV + MODEL
# =========================
env = TrafficEnv(grid_size=3)
model = PPO.load("ppo_city")

state, _ = env.reset()

clock = pygame.time.Clock()

CELL_SIZE = 150
OFFSET_X = 100
OFFSET_Y = 100


# =========================
# 🚗 DRAW FUNCTION
# =========================
def draw_city(env, action):
    sim = env.sim

    for r in range(sim.grid_size):
        for c in range(sim.grid_size):
            x = OFFSET_X + c * CELL_SIZE
            y = OFFSET_Y + r * CELL_SIZE

            # Draw intersection box
            pygame.draw.rect(screen, (50, 50, 50), (x, y, CELL_SIZE, CELL_SIZE), 2)

            # Draw vehicles
            count = int(sim.get_state()[r][c])
            for i in range(count):
                pygame.draw.circle(
                    screen,
                    (0, 200, 255),
                    (x + 20 + (i % 10) * 10, y + 30 + (i // 10) * 10),
                    4,
                )

            # Draw signal
            signal = sim.signals[r][c]
            color = (0, 255, 0) if signal.state == "NS" else (255, 0, 0)

            pygame.draw.circle(screen, color, (x + CELL_SIZE - 20, y + 20), 10)

            # Highlight active action
            idx = r * sim.grid_size + c
            if idx == action:
                pygame.draw.rect(screen, (255, 255, 0), (x, y, CELL_SIZE, CELL_SIZE), 3)


# =========================
# 🔁 MAIN LOOP
# =========================
running = True

while running:
    screen.fill((0, 0, 0))

    # AI decision
    action, _ = model.predict(state)

    state, reward, done, _, _ = env.step(action)

    # Draw city
    draw_city(env, action)

    # Reward display
    text = font.render(f"Reward: {round(reward, 2)}", True, (255, 255, 255))
    screen.blit(text, (350, 30))

    pygame.display.flip()
    clock.tick(5)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()