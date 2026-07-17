import cv2
from ultralytics import YOLO
import os

# Absolute Pathing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n.pt")

class TrafficVision:
    def __init__(self):
        # Load your industry-standard YOLOv8 model
        self.model = YOLO(MODEL_PATH)
        self.vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck

    def detect_vehicles(self, frame):
        """Processes a single frame and returns the count."""
        results = self.model(frame, verbose=False)
        count = 0
        
        for r in results:
            for box in r.boxes:
                if int(box.cls) in self.vehicle_classes:
                    count += 1
        
        return count, results[0].plot() # Return count and annotated image

if __name__ == "__main__":
    # Test script with your webcam or a video file
    vision = TrafficVision()
    cap = cv2.VideoCapture(0) # 0 for Webcam
    
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            count, annotated_frame = vision.detect_vehicles(frame)
            cv2.putText(annotated_frame, f"Vehicles: {count}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Jagatpura CCTV - AI Perception", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()