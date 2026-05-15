# ai-engine/snn/live_real_detection.py

import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"

import cv2
import json
import time
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from ultralytics import YOLO

# ==========================
# Load YOLO
# ==========================

yolo_model = YOLO("yolov8n.pt")

# Optimize YOLO
yolo_model.fuse()

# ==========================
# SNN Model
# ==========================

class RealViolenceSNN(nn.Module):

    def __init__(self):

        super().__init__()

        beta = 0.9

        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(120, 256)

        self.dropout1 = nn.Dropout(0.3)

        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

        self.fc2 = nn.Linear(256, 128)

        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad
        )

        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Layer 1
        cur1 = self.fc1(x)

        cur1 = self.dropout1(cur1)

        spk1, mem1 = self.lif1(
            cur1,
            mem1
        )

        # Layer 2
        cur2 = self.fc2(spk1)

        spk2, mem2 = self.lif2(
            cur2,
            mem2
        )

        # Output Layer
        output = self.fc3(spk2)

        return output

# ==========================
# Initialize SNN
# ==========================

snn_model = RealViolenceSNN()

# Load trained weights
snn_model.load_state_dict(
    torch.load(
        "models/real_violence_snn.pth",
        map_location=torch.device("cpu")
    )
)

snn_model.eval()

print("\n===== TRAINED SNN LOADED =====")

# ==========================
# Webcam
# ==========================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

# Camera warmup
time.sleep(2)

# Reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ==========================
# Variables
# ==========================

prev_gray = None

prev_motion = 0

feature_sequence = []

frame_count = 0

prediction_text = "COLLECTING DATA..."

color = (0, 255, 255)

# ==========================
# Main Loop
# ==========================

while True:

    ret, frame = cap.read()

    # ==========================
    # Safety Checks
    # ==========================

    if not ret:
        continue

    if frame is None:
        continue

    frame_count += 1

    # ==========================
    # Skip Frames
    # ==========================

    # Process every 8th frame
    if frame_count % 8 != 0:

        cv2.putText(
            frame,
            prediction_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow(
            "NeuroWatch AI",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        continue

    # ==========================
    # YOLO Detection
    # ==========================

    results = yolo_model(
        frame,
        classes=[0],
        verbose=False
    )

    annotated_frame = results[0].plot()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    if prev_gray is None:

        prev_gray = gray

        continue

    # ==========================
    # Optical Flow
    # ==========================

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    # ==========================
    # Feature Extraction
    # ==========================

    motion_mag = np.mean(magnitude)

    motion_var = np.var(magnitude)

    acceleration = abs(
        motion_mag - prev_motion
    )

    direction_change = np.mean(angle)

    prev_motion = motion_mag

    # ==========================
    # Smooth Normalization
    # ==========================

    motion_mag_norm = np.tanh(
        motion_mag / 20
    )

    motion_var_norm = np.tanh(
        motion_var / 50
    )

    acceleration_norm = np.tanh(
        acceleration / 10
    )

    direction_norm = np.tanh(
        direction_change / 5
    )

    # ==========================
    # Build Feature Vector
    # ==========================

    feature_vector = [

        motion_mag_norm,

        motion_var_norm,

        acceleration_norm,

        direction_norm
    ]

    feature_sequence.extend(
        feature_vector
    )

    # Keep latest 40 values
    if len(feature_sequence) > 40:

        feature_sequence = feature_sequence[-40:]

    # ==========================
    # SNN Prediction
    # ==========================

    if len(feature_sequence) == 40:

        # Expand to model input size
        padded_features = (
            feature_sequence * 3
        )[:120]

        input_tensor = torch.tensor(
            [padded_features],
            dtype=torch.float
        )

        with torch.no_grad():

            output = snn_model(
                input_tensor
            )

            prediction = output.argmax(
                dim=1
            )

            # ==========================
            # Violence Detected
            # ==========================

            if prediction.item() == 1:

                prediction_text = (
                    "VIOLENT ACTIVITY DETECTED"
                )

                threat = "HIGH"

                color = (0, 0, 255)

            # ==========================
            # Normal Activity
            # ==========================

            else:

                prediction_text = (
                    "NORMAL ACTIVITY"
                )

                threat = "LOW"

                color = (0, 255, 0)

            # ==========================
            # Update Shared AI State
            # ==========================

            ai_state = {

                "status": prediction_text,

                "threat": threat,

                "fps": 24,

                "detections": len(
                    results[0].boxes
                ),

                "accuracy": 76.62
            }

            with open(
                "ai_state.json",
                "w"
            ) as file:

                json.dump(
                    ai_state,
                    file
                )

    # ==========================
    # Display Prediction
    # ==========================

    cv2.putText(
        annotated_frame,
        prediction_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        annotated_frame,
        f"Motion:{motion_mag:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "NeuroWatch AI",
        annotated_frame
    )

    prev_gray = gray

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# Cleanup
# ==========================

cap.release()

cv2.destroyAllWindows()