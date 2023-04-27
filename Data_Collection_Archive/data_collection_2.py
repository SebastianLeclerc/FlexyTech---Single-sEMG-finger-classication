import random
import time
import matplotlib.pyplot as plt
import keyboard

plt.ion()
fig, axs = plt.subplots(5, 1, sharex=True)
lines = [ax.plot([], [])[0] for ax in axs]

key_to_finger = {'t': 5, 'i': 1, 'm': 2, 'r': 3, 'l': 4}

finger_data = [[], [], [], [], []]
while True:
    for key, finger_idx in key_to_finger.items():
        if keyboard.is_pressed(key):
            data = random.uniform(0, 1)
            finger_data[finger_idx].append(data)
            lines[finger_idx].set_ydata(finger_data[finger_idx])
            axs[finger_idx].relim()
            axs[finger_idx].autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
    time.sleep(1/1000)
