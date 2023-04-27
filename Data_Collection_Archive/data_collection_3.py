import random
import time
import matplotlib.pyplot as plt
import keyboard
import csv

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])

key_to_label = {'a': 'Thumb', 's': 'Index Finger', 'd': 'Middle Finger', 'f': 'Ring Finger', 'g': 'Little Finger'}

filename = 'data.csv'
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Time', 'Value', 'Label'])
    xdata, ydata, labels = [], [], []
    while True:
        if keyboard.is_pressed('a') or keyboard.is_pressed('s') or keyboard.is_pressed('d') or keyboard.is_pressed('f') or keyboard.is_pressed('g'):
            for key in key_to_label:
                if keyboard.is_pressed(key):
                    label = key_to_label[key]
                    data = random.uniform(0, 1)
                    xdata.append(time.time())
                    ydata.append(data)
                    labels.append(label)
                    line.set_xdata(xdata)
                    line.set_ydata(ydata)
                    ax.relim()
                    ax.autoscale_view()
                    for i, txt in enumerate(labels):
                        ax.annotate(txt, (xdata[i], ydata[i]))
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    csvwriter.writerow([xdata[-1], ydata[-1], labels[-1]])
                    time.sleep(1/1000)
