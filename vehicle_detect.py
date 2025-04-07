import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt") 
model.to("mps")  # Apple Metal | Windows: model.to("cuda")

cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Error: Could not open the video!")
    exit()

class_names = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            # Confidence score 
            if cls in class_names and conf > 0.1:
                vehicle_type = class_names[cls]

                # Draw the box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display the object name and confidence score
                label = f"{vehicle_type} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) == 27: # Press esc to exit
        break

cap.release()
cv2.destroyAllWindows()