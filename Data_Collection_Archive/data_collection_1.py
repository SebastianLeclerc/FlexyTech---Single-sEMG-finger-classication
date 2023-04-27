import random
import time
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])

xdata, ydata = [], []
while True:
    data = random.uniform(1, 2)
    xdata.append(time.time())
    ydata.append(data)
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1/1000)
