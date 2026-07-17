import time
import cv2
import numpy as np
from stable_baselines3 import PPO
from env.traffic_env import TrafficEnv

def visualize_agent():
    # 1. Load the Environment and the 9Q Model
    env = TrafficEnv()
    model = PPO.load("ppo_quantum_traffic_9Q_best")

    obs, _ = env.reset()
    done = False
    
    print("🚗 Starting Visualization... Press 'q' to exit.")

    while not done:
        # 2. Get action from the AI
        action, _ = model.predict(obs, deterministic=True)
        
        # 3. Step the environment
        obs, reward, done, truncated, _ = env.step(action)

        # 4. Create a Visual Rendering (Simple OpenCV)
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        grid = env.sim.get_state()
        
        # Draw Intersections & Cars
        for r in range(3):
            for c in range(3):
                x, y = (c + 1) * 100, (r + 1) * 100
                
                # Draw Signal
                signal_state = env.sim.signals[r][c].state
                color = (0, 255, 0) if signal_state == "NS" else (0, 0, 255) # Green for NS, Red for EW
                cv2.rectangle(frame, (x-20, y-20), (x+20, y+20), color, -1)
                
                # Draw Car Count
                cv2.putText(frame, str(int(grid[r][c])), (x-10, y+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Reward: {reward:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("9-Qubit Traffic AI in Action", frame)
        
        # Slow down so we can watch it
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_agent()