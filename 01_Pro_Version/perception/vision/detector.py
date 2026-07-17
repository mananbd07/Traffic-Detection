import cv2
import numpy as np
from ultralytics import YOLO

class TrafficDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load the YOLOv8 model sitting in your root directory
        self.model = YOLO(model_path)
        # Define 9 grid zones (3x3). These are normalized (0.0 to 1.0) 
        # coordinates for a standard landscape camera view.
        self.zones = self._setup_zones()

    def _setup_zones(self):
        # Dividing the frame into 9 equal sectors for the 3x3 grid
        zones = []
        for row in range(3):
            for col in range(3):
                zones.append({
                    'x_range': (col * 0.33, (col + 1) * 0.33),
                    'y_range': (row * 0.33, (row + 1) * 0.33)
                })
        return zones

    def get_densities(self, frame):
        height, width, _ = frame.shape
        results = self.model(frame, verbose=False)
        densities = [0] * 9
        
        # We only care about cars, trucks, and buses (COCO classes 2, 5, 7)
        target_classes = [2, 3, 5, 7] 

        for r in results:
            for box in r.boxes:
                if int(box.cls) in target_classes:
                    # Get center of vehicle
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx_norm = ((x1 + x2) / 2) / width
                    cy_norm = ((y1 + y2) / 2) / height
                    
                    # Assign to zone
                    for i, zone in enumerate(self.zones):
                        if (zone['x_range'][0] <= cx_norm < zone['x_range'][1] and 
                            zone['y_range'][0] <= cy_norm < zone['y_range'][1]):
                            densities[i] += 1
                            break
                            
        # Normalize: 0 to 1.0 (assuming 10 cars is 'full' density)
        return [min(d / 10.0, 1.0) for d in densities]