import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolo11n.pt")
model.to("mps") # Apple Metal | Windows: model.to("cuda")

cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Error: Couldn't open the video!")
    exit()

class_names = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

vehicle_counts = {name: 0 for name in class_names.values()}
tracked_vehicles = {}  

next_id = 1 # Start tracking IDs
distance_threshold = 30 

def get_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, stream=True)

    current_frame_vehicles = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            # Confidence score
            if cls in class_names and conf > 0.2:
                vehicle_type = class_names[cls]
                cx, cy = get_center(x1, y1, x2, y2)

                same_id_found = False
                for v_id, (prev_cx, prev_cy) in tracked_vehicles.items():
                    dist = np.linalg.norm(np.array((cx, cy)) - np.array((prev_cx, prev_cy)))
                    if dist < distance_threshold:
                        same_id_found = True
                        tracked_vehicles[v_id] = (cx, cy) # Update the tracked position
                        vehicle_id = v_id
                        break

                if not same_id_found:
                    vehicle_id = next_id
                    tracked_vehicles[vehicle_id] = (cx, cy) # Track the new vehicle
                    next_id += 1 # Increment the ID counter
                    vehicle_counts[vehicle_type] += 1  # Increment the count for this vehicle type

                # Draw the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{vehicle_type} {conf:.2f} ID:{vehicle_id}" 
                cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the count of each vehicle type
    y_offset = 30
    for v_type, count in vehicle_counts.items():
        text = f"{v_type}: {count}"
        cv2.putText(frame, text, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_offset += 30

    cv2.imshow("Vehicle Detection & Count", frame)

    if cv2.waitKey(1) == 27: # Press esc to exit
        break

cap.release()
cv2.destroyAllWindows()