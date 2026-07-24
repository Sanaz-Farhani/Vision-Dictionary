import cv2
import math
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

CONFIDENCE = 0.50

while True:

    success, frame = cap.read()

    if not success:
        break

    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2

    results = model(frame, verbose=False)

    selected_box = None
    shortest_distance = float("inf")

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            object_center_x = (x1 + x2) // 2
            object_center_y = (y1 + y2) // 2

            distance = math.hypot(
                object_center_x - center_x,
                object_center_y - center_y
            )

            if distance < shortest_distance:
                shortest_distance = distance
                selected_box = box

    if selected_box is not None:

        x1, y1, x2, y2 = map(int, selected_box.xyxy[0])

        class_id = int(selected_box.cls[0])
        label = model.names[class_id].capitalize()

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Live Camera Dictionary", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()