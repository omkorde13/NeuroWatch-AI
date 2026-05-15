# ai-engine/training/extract_motion_spikes.py

import cv2
import numpy as np
import os
from tqdm import tqdm

# ==========================
# Dataset Paths
# ==========================

VIOLENCE_PATH = "datasets/violence-dataset/Violence"
NONVIOLENCE_PATH = "datasets/violence-dataset/NonViolence"

# ==========================
# Storage
# ==========================

X = []
y = []

# ==========================
# Parameters
# ==========================

# Increased temporal window
MAX_SEQUENCE_LENGTH = 120

# ==========================
# Process Video Function
# ==========================

def process_video(video_path, label):

    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()

    if not ret:
        return

    prev_gray = cv2.cvtColor(
        prev_frame,
        cv2.COLOR_BGR2GRAY
    )

    prev_motion = 0

    spike_sequence = []

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Process every 3rd frame
        if frame_count % 3 != 0:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

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
        # Smooth Feature Scaling
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

        # Multi-feature spike vector
        spike_vector = [

            motion_mag_norm,

            motion_var_norm,

            acceleration_norm,

            direction_norm
        ]

        spike_sequence.extend(spike_vector)

        prev_gray = gray

        # Stop once enough sequence collected
        if len(spike_sequence) >= MAX_SEQUENCE_LENGTH:
            break

    cap.release()

    # ==========================
    # Save Valid Sequence
    # ==========================

    if len(spike_sequence) >= MAX_SEQUENCE_LENGTH:

        spike_sequence = spike_sequence[:MAX_SEQUENCE_LENGTH]

        X.append(spike_sequence)

        y.append(label)

# ==========================
# Process Violence Videos
# ==========================

print("\n===== Processing Violence Videos =====\n")

for video in tqdm(os.listdir(VIOLENCE_PATH)):

    process_video(
        os.path.join(VIOLENCE_PATH, video),
        1
    )

# ==========================
# Process NonViolence Videos
# ==========================

print("\n===== Processing NonViolence Videos =====\n")

for video in tqdm(os.listdir(NONVIOLENCE_PATH)):

    process_video(
        os.path.join(NONVIOLENCE_PATH, video),
        0
    )

# ==========================
# Convert Arrays
# ==========================

X = np.array(X)
y = np.array(y)

# ==========================
# Save Dataset
# ==========================

os.makedirs(
    "datasets/processed",
    exist_ok=True
)

np.save(
    "datasets/processed/X.npy",
    X
)

np.save(
    "datasets/processed/y.npy",
    y
)

# ==========================
# Final Output
# ==========================

print("\n===== ADVANCED DATASET READY =====")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nSaved Files:")
print("datasets/processed/X.npy")
print("datasets/processed/y.npy")