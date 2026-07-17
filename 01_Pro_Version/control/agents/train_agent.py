import os
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from huggingface_hub import HfApi
from env.traffic_env import TrafficEnv

# --- MANANBD07 9-QUBIT AUTOPILOT CALLBACK ---
class QuantumAutopilotCallback(BaseCallback):
    def __init__(self, check_freq=10000, repo_id="MananBd07/Traffic_AI_Quantum_9Q"):
        super().__init__()
        self.check_freq = check_freq
        self.best_mean_reward = -float('inf')
        self.repo_id = repo_id
        self.api = HfApi()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum([ep['r'] for ep in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"\n💎 [New Peak] 9-Qubit Reward: {mean_reward:.2f}. Syncing to MananBd07 Cloud...")
                    self.model.save("ppo_quantum_traffic_9Q_best")
                    try:
                        self.api.upload_file(
                            path_or_fileobj="ppo_quantum_traffic_9Q_best.zip",
                            path_in_repo="model_9Q_best.zip",
                            repo_id=self.repo_id
                        )
                    except Exception as e:
                        print(f"Sync error: {e}")
        return True

def run_manan_marathon_9Q():
    # 1. Create a fresh environment
    env = TrafficEnv()
    
    # 2. Setup the 9-Qubit Bottleneck Architecture
    # pi=[128, 9, 128] means 128 neurons compress to 9 "Quantum States", then expand back to 128
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 9, 128], vf=[128, 128]),
        activation_fn=torch.nn.Tanh
    )

    # 3. Initialize PPO with RTX 3050 Optimized Hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=2e-4,    # Slightly lower for more complex 9-qubit landscape
        n_steps=4096,          
        batch_size=256,        
        n_epochs=12,           # Increased epochs for deeper optimization
        gamma=0.995,           
        ent_coef=0.04,         # High exploration to handle the 9-qubit complexity
        tensorboard_log="./training_logs/",
        device="cuda"          # RTX 3050
    )

    print("-" * 50)
    print(f"⚡ [Status] STARTING 9-QUBIT 5,000,000 STEP MARATHON")
    print(f"👤 [User] MananBd07 | [GPU] RTX 3050")
    print(f"🕒 START TIME: {time.ctime()}")
    print("-" * 50)
    
    try:
        model.learn(
            total_timesteps=5000000, 
            callback=QuantumAutopilotCallback(repo_id="MananBd07/Traffic_AI_Quantum_9Q"),
            progress_bar=True
        )
        model.save("Manan_9Q_Final_Model")
        print(f"\n🏁 9-QUBIT MARATHON COMPLETE.")
    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted. Saving current progress...")
        model.save("Manan_9Q_Emergency_Save")

if __name__ == "__main__":
    # Ensure the 9Q repo exists before starting
    from huggingface_hub import create_repo
    try:
        create_repo("MananBd07/Traffic_AI_Quantum_9Q", exist_ok=True)
    except:
        pass

    run_manan_marathon_9Q()