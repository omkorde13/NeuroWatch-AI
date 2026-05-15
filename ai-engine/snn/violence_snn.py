import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import matplotlib.pyplot as plt

# ==========================
# Synthetic Training Data
# ==========================

# 0 = normal
# 1 = violence

data = torch.tensor([

    # Normal motion
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],

    # Violent motion
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],

    # Normal
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],

    # Violent
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]

], dtype=torch.float)

labels = torch.tensor([
    0,
    1,
    0,
    1
])

# ==========================
# SNN Model
# ==========================

class ViolenceSNN(nn.Module):

    def __init__(self):

        super().__init__()

        beta = 0.9

        spike_grad = surrogate.fast_sigmoid()

        # Fully connected layers
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

        # Initialize membrane states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Layer 1
        cur1 = self.fc1(x)

        spk1, mem1 = self.lif1(cur1, mem1)

        # Layer 2
        cur2 = self.fc2(spk1)

        spk2, mem2 = self.lif2(cur2, mem2)

        return spk2

# ==========================
# Initialize Model
# ==========================

model = ViolenceSNN()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# ==========================
# Training
# ==========================

loss_history = []

print("\n===== TRAINING STARTED =====\n")

for epoch in range(100):

    model.train()

    optimizer.zero_grad()

    # Forward pass
    output = model(data)

    # Compute loss
    loss = loss_fn(output, labels)

    # Backpropagation
    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch} | "
            f"Loss: {loss.item():.4f}"
        )

print("\n===== TRAINING COMPLETE =====\n")

# ==========================
# Testing
# ==========================

print("\n===== TESTING =====\n")

# Test spike pattern
test_data = torch.tensor([
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
], dtype=torch.float)

with torch.no_grad():

    test_output = model(test_data)

    prediction = test_output.argmax(dim=1)

    if prediction.item() == 1:

        print("VIOLENT ACTIVITY DETECTED")

    else:

        print("NORMAL ACTIVITY")

# ==========================
# Plot Training Loss
# ==========================

plt.figure(figsize=(10, 5))

plt.plot(
    loss_history,
    linewidth=2
)

plt.title("SNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid()

plt.show()

print("\n===== NeuroWatch SNN Ready =====")