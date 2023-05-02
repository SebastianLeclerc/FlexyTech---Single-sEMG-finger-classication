import pandas as pd
import json
import matplotlib.pyplot as plt

# This is just a test implementation.
# We could adapt it to generate plots for all JSON files in the directory ?
df = pd.read_json('../data/Tori_test.json')

label = df['label'][0]
x = df['data']
time = df['time']

fig, ax = plt.subplots()
ax.plot(time, x)
ax.set_title(label)
ax.set_xlabel('Time')
ax.set_ylabel('Data')
ax.grid()

plt.savefig('../plots/Tori_test.png')