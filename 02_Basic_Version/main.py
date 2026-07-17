from env.traffic_env import TrafficEnv
from stable_baselines3 import PPO

env = TrafficEnv()
model = PPO.load("ppo_traffic")

obs, _ = env.reset()

for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

    print("State:", obs)
    print("Reward:", reward)

    if done:
        break