# ai-engine/training/train_real_snn.py

import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ==========================
# Load Dataset
# ==========================

X = np.load("datasets/processed/X.npy")
y = np.load("datasets/processed/y.npy")

print("\nDataset Loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================
# Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Convert to Tensors
# ==========================

X_train = torch.tensor(
    X_train,
    dtype=torch.float
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)

# ==========================
# Advanced SNN Model
# ==========================

class RealViolenceSNN(nn.Module):

    def __init__(self):

        super().__init__()

        beta = 0.9

        spike_grad = surrogate.fast_sigmoid()

        # Larger network
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
# Initialize Model
# ==========================

model = RealViolenceSNN()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# ==========================
# Training
# ==========================

loss_history = []

epochs = 600

print("\n===== TRAINING STARTED =====\n")

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    output = model(X_train)

    loss = loss_fn(
        output,
        y_train
    )

    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch} | "
            f"Loss: {loss.item():.4f}"
        )

print("\n===== TRAINING COMPLETE =====")

# ==========================
# Testing
# ==========================

model.eval()

with torch.no_grad():

    predictions = model(X_test)

    predicted_classes = predictions.argmax(dim=1)

    accuracy = accuracy_score(
        y_test.numpy(),
        predicted_classes.numpy()
    )

print(
    f"\nTest Accuracy: "
    f"{accuracy * 100:.2f}%"
)

# ==========================
# Save Model
# ==========================

os.makedirs(
    "models",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "models/real_violence_snn.pth"
)

print("\nModel Saved:")
print("models/real_violence_snn.pth")

# ==========================
# Plot Training Loss
# ==========================

plt.figure(figsize=(10, 5))

plt.plot(
    loss_history,
    linewidth=2
)

plt.title("Advanced SNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid()

plt.show()

print("\n===== NeuroWatch ADVANCED SNN READY =====")