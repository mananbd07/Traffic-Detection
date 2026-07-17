import pygame
import numpy as np
from stable_baselines3 import PPO
from env.traffic_env import TrafficEnv

# Initialize
pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Traffic Control")

font = pygame.font.SysFont(None, 24)

env = TrafficEnv()
model = PPO.load("ppo_traffic")

state, _ = env.reset()

clock = pygame.time.Clock()


def draw_intersection(x_offset, lanes, active_lane, label):
    # 🔥 Intersection label
    title = font.render(label, True, (255, 255, 255))
    screen.blit(title, (x_offset, 60))

    total_cars = int(np.sum(lanes))

    for i in range(len(lanes)):
        y = 100 + i * 80

        # Draw lane box
        pygame.draw.rect(screen, (50, 50, 50), (x_offset, y, 200, 50))

        # Draw cars
        for c in range(int(lanes[i])):
            pygame.draw.circle(
                screen,
                (0, 200, 255),
                (x_offset + 20 + c * 10, y + 25),
                5
            )

        # Draw signal
        color = (0, 255, 0) if i == active_lane else (255, 0, 0)
        pygame.draw.circle(screen, color, (x_offset + 220, y + 25), 10)

        # Lane label
        text = font.render(f"Lane {i}", True, (255, 255, 255))
        screen.blit(text, (x_offset, y - 20))

    # 🔥 Total cars display
    load_text = font.render(f"Cars: {total_cars}", True, (200, 200, 200))
    screen.blit(load_text, (x_offset, 450))


running = True

while running:
    screen.fill((0, 0, 0))

    # Get action from model
    action, _ = model.predict(state)
    state, reward, done, _, _ = env.step(action)

    lanes_A = env.sim.lanes_A
    lanes_B = env.sim.lanes_B

    action_A = int(action[0])
    action_B = int(action[1])

    # Draw intersections
    draw_intersection(100, lanes_A, action_A, "Intersection A")
    draw_intersection(450, lanes_B, action_B, "Intersection B")

    # 🔥 Reward display
    reward_text = font.render(f"Reward: {round(reward, 2)}", True, (255, 255, 255))
    screen.blit(reward_text, (300, 20))

    # 🔥 AI decision display
    action_text = font.render(f"AI Action → A: {action_A} | B: {action_B}", True, (255, 255, 0))
    screen.blit(action_text, (250, 50))

    pygame.display.flip()

    # 🔥 Slower for demo clarity
    clock.tick(2)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()