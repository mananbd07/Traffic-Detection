import importlib.util
import os

def check_dependencies():
    packages = ["ultralytics", "cv2", "stable_baselines3", "torch"]
    missing = []
    
    print("--- Checking Libraries ---")
    for package in packages:
        spec = importlib.util.find_spec(package if package != "cv2" else "cv2")
        if spec is None:
            print(f"❌ {package} is MISSING")
            missing.append(package)
        else:
            print(f"✅ {package} is installed")
    return missing

def check_files():
    required_files = ["yolov8n.pt", "ppo_quantum_traffic_9Q_best.zip"]
    missing_files = []
    
    print("\n--- Checking Project Files ---")
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} is MISSING from the current folder")
            missing_files.append(file)
    return missing_files

if __name__ == "__main__":
    missing_pkgs = check_dependencies()
    missing_fls = check_files()
    
    if not missing_pkgs and not missing_fls:
        print("\n🚀 ALL SYSTEMS GO! You are ready to run main.py.")
    else:
        print("\n⚠️ ACTION REQUIRED:")
        if missing_pkgs:
            print(f"Install libraries: pip install {' '.join(['opencv-python' if p=='cv2' else p for p in missing_pkgs])}")
        if missing_fls:
            print("Move the missing .pt or .zip files into this folder.")