from ultralytics import YOLO
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Tracking history
track_history = {}

# Spike threshold
SPIKE_THRESHOLD = 10

# Store spikes
spike_history = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0]
    )

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    current_spikes = []

    if boxes.id is not None:

        track_ids = boxes.id.int().cpu().tolist()
        xywh = boxes.xywh.cpu()

        for box, track_id in zip(xywh, track_ids):

            x, y, w, h = box

            center_x = int(x)
            center_y = int(y)

            current_position = (center_x, center_y)

            if track_id not in track_history:
                track_history[track_id] = []

            track = track_history[track_id]
            track.append(current_position)

            if len(track) > 20:
                track.pop(0)

            velocity = 0

            if len(track) >= 2:

                x1, y1 = track[-2]
                x2, y2 = track[-1]

                velocity = math.sqrt(
                    (x2 - x1) ** 2 +
                    (y2 - y1) ** 2
                )

            # Spike Encoding
            spike = 1 if velocity > SPIKE_THRESHOLD else 0

            current_spikes.append(spike)

            # Draw spike status
            cv2.putText(
                annotated_frame,
                f"Spike:{spike}",
                (center_x, center_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            # Draw speed
            cv2.putText(
                annotated_frame,
                f"Speed:{int(velocity)}",
                (center_x, center_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    spike_history.append(current_spikes)

    cv2.imshow(
        "NeuroWatch Spike Encoder",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print spike history
print("\nSpike History:")
for spikes in spike_history[:20]:
    print(spikes)