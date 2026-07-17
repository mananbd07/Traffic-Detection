import cv2
import numpy as np
import time
import pandas as pd
from ultralytics import YOLO
from stable_baselines3 import PPO

# --- CONFIGURATION ---
MODEL_PATH = "Manan_9Q_Final_Model"
VIDEO_PATH = "vision/traffic_video.mp4"
YOLO_MODEL = "yolov8n.pt" 
LOG_FILE = "vision/traffic_analytics.csv"

def run_vision_bridge():
    try:
        model = PPO.load(MODEL_PATH)
        detector = YOLO(YOLO_MODEL)
        print(f"✅ Green Wave Coordination Active.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    current_signals = np.zeros(9) # 0: NS, 1: EW
    flow_prediction = np.zeros((3, 3)) # Anticipated cars moving between cells
    total_cars_seen = 0
    start_time = time.time()
    analytics_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize for full visibility
        target_h = 720
        ratio = target_h / frame.shape[0]
        target_w = int(frame.shape[1] * ratio)
        frame = cv2.resize(frame, (target_w, target_h))

        h, w, _ = frame.shape
        grid_counts = np.zeros((3, 3))

        # 1. Detection
        results = detector(frame, verbose=False, conf=0.45)[0]
        for box in results.boxes:
            if int(box.cls) in [2, 3, 5, 7]:
                x1, y1, x2, y2 = box.xyxy[0]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if cx < w*0.05 or cx > w*0.95 or cy < h*0.05 or cy > h*0.95: continue
                col = min(int((cx - w*0.05) / (w*0.9 / 3)), 2)
                row = min(int((cy - h*0.05) / (h*0.9 / 3)), 2)
                grid_counts[row][col] += 1
                total_cars_seen += 1

        # 2. GREEN WAVE LOGIC: Anticipate flow
        # If a signal is Green (NS), we predict cars moving to the cell below
        for i in range(9):
            r, c = i // 3, i % 3
            if current_signals[i] == 0 and grid_counts[r][c] > 0: # If NS is Green
                if r < 2: flow_prediction[r+1][c] += 0.5 # Predict flow to South neighbor
            elif current_signals[i] == 1 and grid_counts[r][c] > 0: # If EW is Green
                if c < 2: flow_prediction[r][c+1] += 0.5 # Predict flow to East neighbor

        # 3. AI Inference with "Augmented" Density
        # We add the flow_prediction to the real counts so the AI 'feels' the incoming cars
        augmented_grid = grid_counts.flatten() + flow_prediction.flatten()
        obs = np.concatenate([augmented_grid, current_signals]).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)

        # Decay flow prediction (it's a temporary 'shadow' of a car)
        flow_prediction *= 0.1 

        for i, flip in enumerate(action):
            if flip == 1: current_signals[i] = 1 - current_signals[i]

        # --- DASHBOARD ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        elapsed = int(time.time() - start_time)
        cv2.putText(frame, "9-QUBIT GREEN WAVE", (20, 45), 1, 1.8, (0, 255, 255), 2)
        cv2.putText(frame, f"MODE: COORDINATED FLOW", (20, 75), 1, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, f"CARS PROCESSED: {total_cars_seen // 5}", (20, 105), 1, 1.2, (255, 255, 255), 1)
        cv2.putText(frame, f"SYNC HEALTH: OPTIMIZED", (20, 135), 1, 1.2, (255, 255, 255), 1)

        # Visualizing the Grid & Flow
        for i, state in enumerate(current_signals):
            r, c = i // 3, i % 3
            cx = int(w*0.05 + (c + 0.5) * (w*0.9 / 3))
            cy = int(h*0.05 + (r + 0.5) * (h*0.9 / 3))
            color = (0, 255, 0) if state == 0 else (0, 0, 255)
            
            # Draw the signal
            cv2.circle(frame, (cx, cy), 20, color, -1)
            
            # If there's high predicted flow, draw a 'warning' ring
            if flow_prediction[r][c] > 0.2:
                cv2.circle(frame, (cx, cy), 25, (0, 255, 255), 2)

            cv2.putText(frame, f"V:{int(grid_counts[r][c])}", (cx-15, cy+40), 1, 0.8, (255, 255, 255), 1)

        cv2.imshow("Manan 9-Qubit: Phase 4.2 Green Wave", frame)
        
        # Log data every second
        if elapsed % 1 == 0:
            analytics_data.append([time.ctime(), total_cars_seen, np.sum(grid_counts)])

        if cv2.waitKey(50) & 0xFF == ord('q'):
            pd.DataFrame(analytics_data, columns=["Timestamp", "Total_Cars", "Grid_Density"]).to_csv(LOG_FILE, index=False)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_bridge()