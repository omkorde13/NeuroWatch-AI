from ultralytics import YOLO
import cv2
import time
import math

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Store tracking history
track_history = {}
velocity_history = {}

# FPS timer
prev_time = 0

# Aggression thresholds
PROXIMITY_THRESHOLD = 200
AGGRESSION_SPEED = 10

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    # Run tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0]  # Detect persons only
    )

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    people_positions = []

    # Check detections exist
    if boxes.id is not None:

        track_ids = boxes.id.int().cpu().tolist()
        xywh = boxes.xywh.cpu()

        for box, track_id in zip(xywh, track_ids):

            x, y, w, h = box

            center_x = int(x)
            center_y = int(y)

            current_position = (center_x, center_y)

            people_positions.append(
                (track_id, center_x, center_y)
            )

            # Initialize track history
            if track_id not in track_history:
                track_history[track_id] = []

            track = track_history[track_id]

            track.append(current_position)

            # Keep last 20 positions
            if len(track) > 20:
                track.pop(0)

            velocity = 0

            # Velocity calculation
            if len(track) >= 2:

                x1, y1 = track[-2]
                x2, y2 = track[-1]

                distance = math.sqrt(
                    (x2 - x1) ** 2 +
                    (y2 - y1) ** 2
                )

                velocity = distance

            velocity_history[track_id] = velocity

            # Draw movement trail
            for i in range(1, len(track)):

                cv2.line(
                    annotated_frame,
                    track[i - 1],
                    track[i],
                    (0, 255, 255),
                    2
                )

            # Show speed
            cv2.putText(
                annotated_frame,
                f"Speed:{int(velocity)}",
                (center_x, center_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    # ==========================
    # Aggression Detection Logic
    # ==========================

    for i in range(len(people_positions)):

        id1, x1, y1 = people_positions[i]

        for j in range(i + 1, len(people_positions)):

            id2, x2, y2 = people_positions[j]

            # Distance between two people
            distance = math.sqrt(
                (x2 - x1) ** 2 +
                (y2 - y1) ** 2
            )

            speed1 = velocity_history.get(id1, 0)
            speed2 = velocity_history.get(id2, 0)

            # Aggression condition
            if (
                distance < PROXIMITY_THRESHOLD and
                (
                    speed1 > AGGRESSION_SPEED or
                    speed2 > AGGRESSION_SPEED
                )
            ):

                print(
                    f"Aggression Triggered | "
                    f"Distance: {distance:.2f} | "
                    f"Speed1: {speed1:.2f} | "
                    f"Speed2: {speed2:.2f}"
                )

                # Warning text
                cv2.putText(
                    annotated_frame,
                    "AGGRESSIVE ACTIVITY",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # Red circles on people
                cv2.circle(
                    annotated_frame,
                    (x1, y1),
                    20,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    annotated_frame,
                    (x2, y2),
                    20,
                    (0, 0, 255),
                    -1
                )

                # Red line between them
                cv2.line(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

    # FPS Calculation
    current_time = time.time()

    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS:{int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show output
    cv2.imshow(
        "NeuroWatch Aggression Detection",
        annotated_frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()