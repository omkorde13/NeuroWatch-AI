import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"

from ultralytics import YOLO
import cv2
import math
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# ==========================
# Load YOLO
# ==========================

model_yolo = YOLO("yolov8n.pt")

# ==========================
# SNN Model
# ==========================

class ViolenceSNN(nn.Module):

    def __init__(self):

        super().__init__()

        beta = 0.9

        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(10, 20)

        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

        self.fc2 = nn.Linear(20, 2)

        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        cur1 = self.fc1(x)

        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.fc2(spk1)

        spk2, mem2 = self.lif2(cur2, mem2)

        return spk2

# ==========================
# Initialize SNN
# ==========================

model_snn = ViolenceSNN()

# ==========================
# Webcam
# ==========================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

track_history = {}

# Store live spike sequence
live_spikes = []

SPIKE_THRESHOLD = 10

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model_yolo.track(
        frame,
        persist=True,
        classes=[0]
    )

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    current_spike = 0

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

            current_spike = spike

            # Display info
            cv2.putText(
                annotated_frame,
                f"Speed:{int(velocity)}",
                (center_x, center_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"Spike:{spike}",
                (center_x, center_y - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # ==========================
    # Build Spike Sequence
    # ==========================

    live_spikes.append(current_spike)

    # Keep last 10 spikes
    if len(live_spikes) > 10:
        live_spikes.pop(0)

    # ==========================
    # SNN Prediction
    # ==========================

    if len(live_spikes) == 10:

        input_tensor = torch.tensor(
            [live_spikes],
            dtype=torch.float
        )

        with torch.no_grad():

            output = model_snn(input_tensor)

            prediction = output.argmax(dim=1)

            if prediction.item() == 1:

                cv2.putText(
                    annotated_frame,
                    "VIOLENT ACTIVITY DETECTED",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            else:

                cv2.putText(
                    annotated_frame,
                    "NORMAL ACTIVITY",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

    cv2.imshow(
        "NeuroWatch Live SNN Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()