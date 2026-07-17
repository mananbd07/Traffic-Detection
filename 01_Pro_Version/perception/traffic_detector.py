import carla
import cv2
import numpy as np
import time
import json
import os
from ultralytics import YOLO

# 1. Load the AI Vision Model
print("🧠 Loading AI Vision Model...")
model = YOLO('yolov8n.pt') 

# Global variable to pass frames safely to the main thread
latest_frame = None

# --- Phase 3: Spatial Mapping Logic ---
def get_zone_counts(results):
    """
    Maps YOLO detections to 4 Spatial Quadrants (North, South, East, West).
    """
    quadrants = {"North": 0, "South": 0, "East": 0, "West": 0}
    
    for r in results:
        for box in r.boxes:
            # Get box center coordinates
            xyxy = box.xyxy[0]
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            
            # Center of 640x480 screen is (320, 240)
            dx = cx - 320
            dy = cy - 240
            
            # Determine which quadrant the car is in based on diagonals
            if abs(dx) > abs(dy):
                if dx > 0:
                    quadrants["East"] += 1
                else:
                    quadrants["West"] += 1
            else:
                if dy > 0:
                    quadrants["South"] += 1
                else:
                    quadrants["North"] += 1
                    
    # Return a 4-feature state vector representing the lane densities
    state_vector = [quadrants["North"], quadrants["South"], quadrants["East"], quadrants["West"]]
    return state_vector

def process_img(image):
    # Convert CARLA raw sensor data to OpenCV format
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    frame = i2[:, :, :3] 

    # 2. Run AI Detection (verbose=False prevents terminal spam, device='cpu' saves VRAM)
    results = list(model(frame, stream=True, verbose=False, device='cpu'))
    
    # 3. Save State for the Brain
    state_vector = get_zone_counts(results)
    current_state = {
        "timestamp": time.time(),
        "densities": state_vector,
        "signals": [0, 0, 0, 0] # 4 signal states for the 4 directions
    }
    
    try:
        backend_path = os.path.join(os.path.dirname(__file__), "..", "data", "state.json")
        with open(backend_path, "w") as f:
            json.dump(current_state, f)
    except Exception as e:
        print(f"⚠️ Bridge Error: {e}")

    # 4. Save the annotated frame for the main thread to display
    global latest_frame
    for r in results:
        annotated_frame = r.plot() 
        
        # --- Phase 4: Augmented Reality Overlays ---
        h, w = annotated_frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw Quadrant Lines (Diagonal Crosshairs)
        cv2.line(annotated_frame, (0, 0), (w, h), (0, 255, 255), 2)
        cv2.line(annotated_frame, (w, 0), (0, h), (0, 255, 255), 2)
        
        # Draw Labels
        cv2.putText(annotated_frame, "NORTH", (center_x - 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(annotated_frame, "SOUTH", (center_x - 30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(annotated_frame, "WEST", (30, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(annotated_frame, "EAST", (w - 100, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        latest_frame = annotated_frame

# --- 5. Persistent Connection & Map Optimization ---
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0) # Increased timeout to allow for map loading

world = None
while world is None:
    try:
        print("🔍 Scanning for CARLA Simulator...")
        world = client.get_world()
        print("✅ AI Vision Linked to Digital Twin.")
        
        # Phase 5: Enforce Lightweight Map
        current_map = world.get_map().name
        if 'Town02' not in current_map:
            print(f"⚠️ Heavy map detected ({current_map}).")
            print("⏳ Switching to lightweight Town02 to prevent RTX 3050 VRAM crashes...")
            world = client.load_world('Town02')
            print("✅ Town02 loaded successfully!")
            
    except RuntimeError:
        print("⌛ Simulator is busy or initializing. Retrying in 2 seconds...")
        time.sleep(2)

# 6. Setup the "Eyes" (Camera Sensor)
blueprint_library = world.get_blueprint_library()
cam_bp = blueprint_library.find('sensor.camera.rgb')

cam_bp.set_attribute('image_size_x', '640')
cam_bp.set_attribute('image_size_y', '480')
# Reduced sensor frequency to 1 frame per second to protect the RTX 3050 GPU
cam_bp.set_attribute('sensor_tick', '1.0') 

# Phase 3: Auto-target Intersection
traffic_lights = world.get_actors().filter('traffic.traffic_light')
if len(traffic_lights) > 0:
    # Pick a traffic light as our intersection anchor
    tl = traffic_lights[0] 
    tl_loc = tl.get_transform().location
    print(f"🎯 Auto-targeting intersection at {tl_loc.x:.1f}, {tl_loc.y:.1f}")
    # Drone view: 40 meters straight up, looking down
    spawn_point = carla.Transform(
        carla.Location(x=tl_loc.x, y=tl_loc.y, z=40),
        carla.Rotation(pitch=-90, yaw=0, roll=0)
    )
else:
    print("⚠️ No traffic lights found for auto-targeting. Using fallback location.")
    spawn_point = carla.Transform(carla.Location(x=-60, y=10, z=40), carla.Rotation(pitch=-90))

# --- Phase 4: Sync CARLA Spectator Camera ---
# Teleporting to Z=40 in Town10HD caused massive VRAM spikes on the RTX 3050.
# Instead, we will place your main CARLA window view safely on the ground at the intersection!
spectator = world.get_spectator()
ground_view = carla.Transform(
    carla.Location(x=tl_loc.x, y=tl_loc.y, z=5.0),
    carla.Rotation(pitch=-15, yaw=0, roll=0)
)
spectator.set_transform(ground_view)
print("🎥 Synced CARLA Digital Twin camera to ground-level at intersection.")

sensor = world.spawn_actor(cam_bp, spawn_point)
sensor.listen(lambda data: process_img(data))

print("🚀 AI Vision System is live. Exporting state to data...")

try:
    cv2.namedWindow("Digital Twin - AI Vision System", cv2.WINDOW_NORMAL)
    while True:
        try:
            # Display frame safely in the main thread
            if latest_frame is not None:
                cv2.imshow("Digital Twin - AI Vision System", latest_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Completely decoupled from CARLA's internal ticking engine to prevent timeouts
            time.sleep(0.1) 
        except Exception as e:
            time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Shutting down AI Vision System...")
    if sensor is not None:
        sensor.destroy()
    cv2.destroyAllWindows()