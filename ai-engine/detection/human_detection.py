from ultralytics import YOLO
import cv2
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

prev_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Track objects
    results = model.track(
        frame,
        persist=True,
        classes=[0]  # Only detect persons
    )

    # Annotated frame
    annotated_frame = results[0].plot()

    # FPS calculation
    current_time = time.time()

    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("NeuroWatch Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()