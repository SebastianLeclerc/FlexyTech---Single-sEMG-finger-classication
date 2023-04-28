import json
import os
import time
import random

fingers = ['thumb', 'index finger', 'middle finger', 'ring finger', 'little finger']
time_to_collect = 10
time_to_rest = 4
participant_name = input("Please enter your name: ")

finger_data = []
for finger in fingers:
    print(f"Please start closing and opening your right {finger}.")
    time.sleep(1)
    data = []
    for i in range(time_to_collect):
        remaining_time = time_to_collect - i
        print(f"{remaining_time} seconds for {finger} remaining...")
        sensor_data = random.uniform(0, 1) # delete this
        data.append(sensor_data)
        time.sleep(1)
        if i == (time_to_collect-1):
            print(f"Collection for {finger} is done. Have a short {time_to_rest} seconds rest now.")
    finger_data.append({"data": data, "finger": finger})
    time.sleep(time_to_rest)

data = {"fingers": finger_data}
filename = f"{participant_name}_{time.strftime('%Y-%m-%d')}.json"
if not os.path.exists("../data"):
    os.makedirs("../data")
file_path = os.path.join("../data", filename)
with open(file_path, 'w') as f:
    json.dump(data, f)

print(f"data collection completed. Thank you, {participant_name}!")
