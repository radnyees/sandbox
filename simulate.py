import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,8,8*500)

brain_signal = np.zeros_like(t)

brain_signal[1000:1050] = 1  # Spike at 2 seconds
brain_signal[2000:2050] = 1  # Spike at 4 seconds
brain_signal[3000:3050] = 1  # Spike at 6 seconds
brain_signal[4000:4050] = 1  # Spike at 8 seconds

plt.plot(t, brain_signal)
plt.title("Simulated Brain Signal: Movement Intention (8 seconds)")
plt.xlabel("Time (s)")
plt.ylabel("Movement Intention (0 or 1)")

plt.savefig("simulated_8secbrainsignal.jpg")
plt.show()

# Process brain signal to detect movement intention
for i, signal in enumerate(brain_signal):
    if signal == 1:
        print(f"Movement intention detected at time {i/500} seconds")
        # Here you would send a command to the electrical stimulation device

# Simulate muscle stimulation
for i, signal in enumerate(brain_signal):
    if signal == 1:
        print(f"Electrical stimulation triggered at time {i/500} seconds to activate muscle.")
