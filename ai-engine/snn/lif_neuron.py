import torch
import snntorch as snn
import matplotlib.pyplot as plt

# Input spike train
input_spikes = torch.tensor([
    0, 0, 1, 0, 1,
    1, 0, 0, 1, 1,
    0, 1, 0, 0, 1,
    1, 1, 0, 1, 0,
    0, 1, 1, 0, 1,
    0, 0, 1, 1, 1,
    0, 0, 1, 0, 1,
    1, 0, 1, 0, 0,
    1, 1, 0, 1, 0
], dtype=torch.float)

# Automatically calculate steps
num_steps = len(input_spikes)

# Create LIF neuron
lif = snn.Leaky(beta=0.9)

# Initialize membrane potential
mem = torch.zeros(1)

# Store outputs
membrane_history = []
spike_history = []

print("\n===== NeuroWatch SNN Simulation =====\n")

# Simulation loop
for step in range(num_steps):

    # Current input spike
    cur_input = input_spikes[step]

    # Run neuron
    spike, mem = lif(cur_input, mem)

    # Store results
    membrane_history.append(mem.item())
    spike_history.append(spike.item())

    # Print neuron state
    print(
        f"Step {step} | "
        f"Input: {cur_input.item()} | "
        f"Membrane: {mem.item():.2f} | "
        f"Spike: {spike.item()}"
    )

# ==========================
# Membrane Potential Plot
# ==========================

plt.figure(figsize=(12, 5))

plt.plot(
    membrane_history,
    linewidth=2
)

plt.title("Membrane Potential Over Time")
plt.xlabel("Time Step")
plt.ylabel("Membrane Potential")

plt.grid()

plt.show()

# ==========================
# Spike Activity Plot
# ==========================

plt.figure(figsize=(12, 3))

spike_times = [
    i for i, spike in enumerate(spike_history)
    if spike == 1
]

plt.eventplot(spike_times)

plt.title("Neuron Spike Activity")
plt.xlabel("Time Step")

plt.show()

print("\n===== Simulation Complete =====")