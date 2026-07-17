from stable_baselines3 import PPO
from env.traffic_env import TrafficEnv


def train_agent():
    env = TrafficEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,   # 🔥 Back to stable
        n_steps=2048,           # 🔥 Not too large
        batch_size=64,
        gamma=0.99,             # 🔥 Stable discount
        gae_lambda=0.95,
        ent_coef=0.02,          # 🔥 Balanced exploration
    )

    model.learn(total_timesteps=200000)

    model.save("ppo_traffic")
    return model