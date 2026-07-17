import carla
import json
import time
import os
import csv

class SmartIntersectionController:
    def __init__(self):
        print("🧠 Initializing Spatial Density Controller (Phase 3)...")
    
    def decide_lights(self, densities):
        """
        Takes 4 densities [North, South, East, West].
        Returns 4 actions [Green/Red] for [North, South, East, West].
        """
        if len(densities) != 4:
            return [0, 0, 0, 0]
            
        north, south, east, west = densities
        ns_total = north + south
        ew_total = east + west
        
        # Whichever axis has more waiting cars gets the Green light
        if ns_total >= ew_total:
            # North/South = Green, East/West = Red
            return [1, 1, 0, 0]
        else:
            # North/South = Red, East/West = Green
            return [0, 0, 1, 1]

def main():
    print("🚀 Starting Smart Master Controller Backend (Terminal 4)")
    
    # 1. CARLA Connection
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    
    world = None
    while world is None:
        try:
            print("🔍 Scanning for CARLA Simulator...")
            world = client.get_world()
            print("✅ Master Controller linked to Digital Twin.")
        except RuntimeError:
            print("⌛ Simulator is busy or initializing. Retrying in 2 seconds...")
            time.sleep(2)

    # 2. Grab Traffic Lights and Group by Intersection
    traffic_lights = []
    print("🚥 Waiting for traffic lights to spawn in the map...")
    while len(traffic_lights) == 0:
        traffic_lights = list(world.get_actors().filter('traffic.traffic_light'))
        if len(traffic_lights) == 0:
            time.sleep(1.0)
            
    print(f"🚦 Found {len(traffic_lights)} traffic lights in the world.")
    
    # Sort lights by distance to the first one so we get the 4 lights at the EXACT same intersection
    center_loc = traffic_lights[0].get_transform().location
    traffic_lights.sort(key=lambda tl: tl.get_transform().location.distance(center_loc))
    intersection_lights = traffic_lights[:4]
    
    # Freeze the simulator's internal traffic manager so it stops overwriting our AI colors
    for tl in intersection_lights:
        tl.freeze(True)
        
    print("🔒 Locked onto intersection and froze default traffic signals.")

    controller = SmartIntersectionController()
    state_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'state.json')
    csv_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'traffic_log.csv')

    # 3. CSV Setup (4 Zones, Human Readable)
    csv_headers = ["Time", "North Cars", "South Cars", "East Cars", "West Cars", 
                   "North Light", "South Light", "East Light", "West Light", 
                   "North Action", "South Action", "East Action", "West Action"]
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(csv_headers)
            print(f"📄 Created new Human-Readable data log file: {csv_file}")

    print(f"📡 Listening for Perception Data at {state_file}...")
    
    last_log_time = 0.0
    last_phase_switch = time.time()
    current_actions = [1, 1, 0, 0] # Default starting phase
    
    try:
        while True:
            if not os.path.exists(state_file):
                time.sleep(0.1)
                continue

            try:
                with open(state_file, 'r') as f:
                    current_state = json.load(f)
                
                densities = current_state.get('densities', [0,0,0,0])
                signals = current_state.get('signals', [0,0,0,0])
                
                # --- Quantum Phase Timer Logic ---
                time_elapsed = time.time() - last_phase_switch
                
                if time_elapsed > 10.0:
                    # Force variation: If a light is green for 10 seconds, force it red to let others pass
                    current_actions = [1 if a == 0 else 0 for a in current_actions]
                    last_phase_switch = time.time()
                elif time_elapsed > 3.0:
                    # After 3 seconds, allow the AI to make a smart decision based on density
                    new_actions = controller.decide_lights(densities)
                    if new_actions != current_actions:
                        current_actions = new_actions
                        last_phase_switch = time.time()
                
                # Apply the actions to the exact intersection lights
                for i, tl in enumerate(intersection_lights):
                    if current_actions[i] == 1:
                        tl.set_state(carla.TrafficLightState.Green)
                    else:
                        tl.set_state(carla.TrafficLightState.Red)
                
                # --- Phase 4: Throttle and Human-Readable Logging ---
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    last_log_time = current_time
                    
                    time_str = time.strftime('%H:%M:%S', time.localtime(current_state.get('timestamp', current_time)))
                    
                    # Convert 0/1 to Red/Green
                    hr_signals = ["Green" if s == 1 else "Red" for s in signals]
                    hr_actions = ["Green" if a == 1 else "Red" for a in current_actions]
                    
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        row = [time_str] + list(densities) + hr_signals + hr_actions
                        writer.writerow(row)
                        
            except json.JSONDecodeError:
                pass
            except PermissionError:
                pass
            except Exception as e:
                print(f"⚠️ Controller Runtime Error: {e}")
                
            time.sleep(0.1) 
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Master Controller...")

if __name__ == "__main__":
    main()
