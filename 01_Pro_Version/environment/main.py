import os
import sys
import traci
import json
import time

# --- 1. DYNAMIC PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_LABS_DIR = os.path.join(ROOT_DIR, "06_Data_Labs")
RESEARCH_DIR = os.path.join(ROOT_DIR, "02_AI_Research")

# Add Research folder to Python Path
sys.path.append(RESEARCH_DIR)
from quantum_logic import calculate_quantum_pressure

# File Paths
NET_FILE = os.path.join(SCRIPT_DIR, "jagatpura.net.xml")
ROU_FILE = os.path.join(SCRIPT_DIR, "routes.rou.xml")
OUTPUT_XML = os.path.join(DATA_LABS_DIR, "results.xml")
OUTPUT_CSV = os.path.join(DATA_LABS_DIR, "results.csv")
LIVE_JSON = os.path.join(DATA_LABS_DIR, "live_status.json")

# --- 2. SUMO ENVIRONMENT ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sumo_path = r"C:\Program Files (x86)\Eclipse\Sumo"
    os.environ['SUMO_HOME'] = sumo_path
    sys.path.append(os.path.join(sumo_path, 'tools'))

def main():
    print("🚀 HYBRID QUANTUM-CLASSIAL ENGINE: STARTING")
    
    sumo_cmd = [
        "sumo-gui", "-n", NET_FILE, "-r", ROU_FILE, 
        "--tripinfo-output", OUTPUT_XML,
        "--delay", "0", "--start", "true", "--quit-on-end", "true"
    ]
    
    traci.start(sumo_cmd)
    tls_ids = traci.trafficlight.getIDList()
    print(f"🚦 Quantum Actuator Monitoring {len(tls_ids)} Signals.")

    step = 0
    while step < 6300:
        traci.simulationStep()
        
        # Dashboard Sync
        if step % 200 == 0:
            try:
                live_snapshot = {
                    "step": step,
                    "active_cars": traci.simulation.getMinExpectedNumber(),
                    "current_congestion": sum(traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList()),
                    "status": "Quantum Optimization Active"
                }
                with open(LIVE_JSON, "w") as f:
                    json.dump(live_snapshot, f)
            except: pass

        # 🧠 QUANTUM AI DECISION LOOP
        if step % 10 == 0:
            for tls in tls_ids:
                lanes = traci.trafficlight.getControlledLanes(tls)
                # Group real-world lane pressure
                q_values = [traci.lane.getLastStepHaltingNumber(l) for l in set(lanes)]
                
                best_phase, confidence = calculate_quantum_pressure(q_values)
                
                # If AI is confident (>70%) about a congestion spike, override lights
                if confidence > 0.7 and sum(q_values) > 5:
                    traci.trafficlight.setPhase(tls, best_phase)
                    traci.trafficlight.setPhaseDuration(tls, 25)

        # Stress Test Injection
        if step % 300 == 0 and step > 0:
            routes = traci.route.getIDList()
            if routes:
                for i in range(20):
                    try: traci.vehicle.add(f"ai_veh_{step}_{i}", routes[0])
                    except: pass
        step += 1

    traci.close()
    
    # Export Data
    print("🏁 Syncing Analytics...")
    xml2csv = os.path.join(os.environ['SUMO_HOME'], 'tools', 'xml', 'xml2csv.py')
    if os.path.exists(OUTPUT_XML):
        os.system(f'python "{xml2csv}" "{OUTPUT_XML}" -o "{OUTPUT_CSV}"')
        print(f"✅ Dashboard Updated.")

if __name__ == "__main__":
    main()