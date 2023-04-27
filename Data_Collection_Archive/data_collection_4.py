import random
import time
import matplotlib.pyplot as plt
import csv
import keyboard

key_to_label = {
    1: 'Thumb',
    2: 'Index Finger',
    3: 'Middle Finger',
    4: 'Ring Finger',
    5: 'Little Finger'
}

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])

xdata, ydata, labeldata = [], [], []
current_label = None

while True:
    if keyboard.is_pressed('esc'):
        break
    for key in key_to_label:
        if keyboard.is_pressed(str(key)) and current_label != key:
            current_label = key
            print('Label:', key_to_label[key])
        elif not keyboard.is_pressed(str(key)) and current_label == key:
            current_label = None
    if current_label is not None:
        data = random.uniform(0, 1)
        xdata.append(time.time())
        ydata.append(data)
        labeldata.append(key_to_label[current_label])
        line.set_xdata(xdata)
        line.set_ydata(ydata)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(1/1000)

filename = 'data.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Value', 'Label'])
    for i in range(len(xdata)):
        writer.writerow([xdata[i], ydata[i], labeldata[i]])
print(f'Data saved to {filename}.')
