from ultralytics import YOLO
import cv2
import time
import math

# Load model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Store previous positions
track_history = {}

prev_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0]
    )

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    if boxes.id is not None:

        track_ids = boxes.id.int().cpu().tolist()
        xywh = boxes.xywh.cpu()

        for box, track_id in zip(xywh, track_ids):

            x, y, w, h = box

            center_x = int(x)
            center_y = int(y)

            current_position = (center_x, center_y)

            # Initialize history
            if track_id not in track_history:
                track_history[track_id] = []

            track = track_history[track_id]
            track.append(current_position)

            # Keep only last 20 positions
            if len(track) > 20:
                track.pop(0)

            # Draw movement path
            for i in range(1, len(track)):
                cv2.line(
                    annotated_frame,
                    track[i - 1],
                    track[i],
                    (0, 255, 255),
                    2
                )

            # Velocity calculation
            if len(track) >= 2:

                x1, y1 = track[-2]
                x2, y2 = track[-1]

                distance = math.sqrt(
                    (x2 - x1) ** 2 +
                    (y2 - y1) ** 2
                )

                velocity = distance

                cv2.putText(
                    annotated_frame,
                    f"Speed: {int(velocity)}",
                    (center_x, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    # FPS
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

    cv2.imshow("NeuroWatch Motion Analysis", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()